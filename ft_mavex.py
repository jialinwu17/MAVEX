# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
import logging
import os
import random
from io import open
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm import tqdm
from bisect import bisect
import yaml, pickle
from easydict import EasyDict as edict
from vqaEval.PythonEvaluationTools.vqaEvalDemo import evaluate_vqa
import pdb, copy, math
import sys, pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_transformers.optimization import (
    AdamW,
    WarmupConstantSchedule,
    WarmupLinearSchedule,
)
from transformers import AutoTokenizer, AutoModel
from vilbert.optimization import RAdam

from torch.optim.lr_scheduler import (
    LambdaLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)

import vilbert.utils as utils
import torch.distributed as dist

from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="pytorch_model_4.bin",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_base_6layer_6conect.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--train_iter_multiplier",
        default=1.0,
        type=float,
        help="multiplier for the multi-task training.",
    )
    parser.add_argument(
        "--train_iter_gap",
        default=4,
        type=int,
        help="forward every n iteration is the validation score is not improving over the last 3 epoch, -1 means will stop",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
             "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
             "0 (default value): dynamic loss scaling.\n"
             "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--save_name", default="", type=str, help="save name for training."
    )
    parser.add_argument(
        "--in_memory",
        default=False,
        type=bool,
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--optim", default="AdamW", type=str, help="what to use for the optimization."
    )
    parser.add_argument(
        "--tasks", default="19", type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--vilbert_freeze",
        default='t10tv5v',
        type=str,
        help="till which layer of textual stream of vilbert need to fixed.",
    )
    parser.add_argument(
        "--knowledge_freeze",
        default='t1t',
        type=str,
        help="till which layer of textual stream of bert need to fixed.",
    )
    parser.add_argument(
        "--vision_scratch",
        action="store_true",
        help="whether pre-trained the image or not.",
    )
    parser.add_argument(
        "--lr_scheduler",
        default="warmup_linear",
        type=str,
        help="whether use learning rate scheduler.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument(
        "--resume_file", default="", type=str, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--dynamic_attention",
        action="store_true",
        help="whether use dynamic attention.",
    )
    parser.add_argument(
        "--clean_train_sets",
        default=True,
        type=bool,
        help="whether clean train sets for multitask data.",
    )
    parser.add_argument(
        "--visual_target",
        default=0,
        type=int,
        help="which target to use for visual branch. \
        0: soft label, \
        1: regress the feature, \
        2: NCE loss.",
    )
    parser.add_argument(
        "--use_wiki",
        default=1,
        type=int,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--use_concept",
        default=1,
        type=int,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--use_image",
        default=1,
        type=int,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--num_wiki_sentences",
        default=50, type=int,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--num_concepts",
        default=80, type=int,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--num_head",
        type=int,
        default=4,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--num_hid",
        type=int,
        default=512,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--loss_vqa",
        default=1, type=float,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--loss_knowledge_pred",
        default=1, type=float,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--mml0",
        default=0.3, type=float,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--bert_size",
        default='tiny', type=str,
        help="whether to use task specific tokens for the multi-task learning.",
    )

    parser.add_argument(
        "--mml1",
        default=0.3, type=float,
        help="whether to use task specific tokens for the multi-task learning.",
    )

    parser.add_argument(
        "--mml2",
        default=1., type=float,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--use_search_word",
        default=1, type=int,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--training_setting",
        default='ft_mavex', type=str,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--new_param_lr",
        default=0.00005, type=float,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--old_param_lr",
        default=0.00002, type=float,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--evaluate",
        default=0, type=int,
        help="whether to use task specific tokens for the multi-task learning.",
    )

    parser.add_argument(
        "--segment_dim",
        default=1024, type=int,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--add_answer_emb",
        default=1, type=int,
        help="whether to use task specific tokens for the multi-task learning.",
    )
    parser.add_argument(
        "--num_epochs",
        default=75, type=int,
        help="whether to use task specific tokens for the multi-task learning.",
    )

    args = parser.parse_args()

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True


    with open("vilbert_tasks.yml", "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    if args.save_name:
        prefix = "-" + args.save_name
    else:
        prefix = ""
    timeStamp = ( args.config_file.split("/")[1].split(".")[0]
            + prefix
    )
    savePath = os.path.join(args.output_dir, timeStamp)

    if args.seed == -1:
        args.seed = int(np.random.randint(100,10000, 1)[0])

    if default_gpu:
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    log_f = open(os.path.join(savePath, 'log.txt'), 'w')
    log_f.writelines(' '.join(sys.argv) + '\n')
    log_f.writelines('Seed: %d '% args.seed + ' '.join(sys.argv) + '\n')
    log_f.flush()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from vilbert.vilbert_mavex import BertConfig

    from vilbert.vilbert_mavex import VILBertForVLTasks
    from vilbert.task_utils import (
        LoadDatasets,
        LoadLosses,
        ForwardModelsTrain,
        ForwardModelsVal)

    task_names = []
    task_lr = []
    for i, task_id in enumerate(args.tasks.split("-")):
        task = "TASK" + task_id
        name = task_cfg[task]["name"]
        task_names.append(name)
        task_lr.append(task_cfg[task]["lr"])

    base_lr = min(task_lr)
    loss_scale = {}
    for i, task_id in enumerate(args.tasks.split("-")):
        task = "TASK" + task_id
        loss_scale[task] = task_lr[i] / base_lr

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    config = BertConfig.from_json_file(args.config_file)
    config.num_wiki_sentences = args.num_wiki_sentences
    config.num_concepts = args.num_concepts
    config.use_wiki = args.use_wiki
    config.use_concept = args.use_concept
    config.use_image = args.use_image
    config.num_head = args.num_head
    config.from_pretrained = args.from_pretrained
    config.segment_dim = args.segment_dim
    config.num_hid = args.num_hid
    config.use_search_word = args.use_search_word
    config.setting = args.training_setting
    config.bert_size = args.bert_size
    config.add_answer_emb = args.add_answer_emb
    config.separate = 0
    if default_gpu:
        # save all the hidden parameters.
        with open(os.path.join(savePath, "command.txt"), "w") as f:
            ddd = vars(args)
            for key in ddd:
                print(key + ': ' + str(ddd[key]), file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, task_dataloader_train, task_dataloader_val = LoadDatasets(
        args, task_cfg, args.tasks.split("-")
    )


    if args.visual_target == 0:
        config.v_target_size = 1601
        config.visual_target = args.visual_target
    else:
        config.v_target_size = 2048
        config.visual_target = args.visual_target

    config.task_specific_tokens = True
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_ave_iter = {}
    task_stop_controller = {}
    for task_id, num_iter in task_num_iters.items():
        task_ave_iter[task_id] = int(
            task_cfg[task]["num_epoch"]
            * num_iter
            * args.train_iter_multiplier
            / args.num_train_epochs
        )
        task_stop_controller[task_id] = utils.MultiTaskStopOnPlateau(
            mode="max",
            patience=1,
            continue_threshold=0.005,
            cooldown=1,
            threshold=0.001,
        )

    task_ave_iter_list = sorted(task_ave_iter.values())
    median_num_iter = task_ave_iter_list[-1]
    num_train_optimization_steps = (
            median_num_iter * args.num_train_epochs // args.gradient_accumulation_steps
    )
    num_labels = max([dataset.num_labels for dataset in task_datasets_train.values()])

    if args.dynamic_attention:
        config.dynamic_attention = True

    model = VILBertForVLTasks.from_pretrained(
        args.from_pretrained,
        config=config,
        num_labels=num_labels,
        default_gpu=default_gpu,
    )
    # bert model from pretyrrained
    if args.add_answer_emb:
        model.answer_embs.init_weight_dict()

    task_losses = LoadLosses(args, task_cfg, args.tasks.split("-"))

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    old_parameters = []
    new_parameters = []
    excluded_parameters = []
    excluded_parameters += ['bert.embeddings.word_embeddings.weight', 'bert.embeddings.position_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight']
    excluded_parameters += ['bert.embeddings.LayerNorm.weight', 'bert.embeddings.LayerNorm.bias', 'bert.embeddings.task_embeddings.weight', 'bert.v_embeddings.image_embeddings.weight']
    excluded_parameters += ['bert.v_embeddings.image_embeddings.bias', 'bert.v_embeddings.image_location_embeddings.weight', 'bert.v_embeddings.image_location_embeddings.bias']
    excluded_parameters += ['bert.v_embeddings.LayerNorm.weight', 'bert.v_embeddings.LayerNorm.bias']

    text_bert_re_init = []
    for key, value in dict(model.named_parameters()).items():
        if (("wiki_" in key) or ("concept_" in key) or ('image_' in key) or ("answer_" in key) or ("mavex" in key) or ("backbone" in key) ) and ('bert.' not in key):
            new_parameters.append(key)
        else:
            old_parameters.append(key)

        if 't' in args.vilbert_freeze and 'bert.encoder.layer.' in key and ('text_bert' not in key):
            if int(key.split('.')[3]) < int(args.vilbert_freeze.split('t')[1]):
                excluded_parameters += [key]
        if 'v' in args.vilbert_freeze and 'bert.encoder.v_layer.' in key and ('text_bert' not in key):
            if int(key.split('.')[3]) < int(args.vilbert_freeze.split('v')[1]):
                excluded_parameters += [key]
        if 'c' in args.vilbert_freeze and 'bert.encoder.c_layer.' in key and ('text_bert' not in key):
            assert 't12t' in args.vilbert_freeze
            assert 'v6v' in args.vilbert_freeze
            if int(key.split('.')[3]) < int(args.vilbert_freeze.split('c')[1]):
                excluded_parameters += [key]

        if 't' in args.knowledge_freeze and 'text_bert.encoder.layer.' in key :
            if int(key.split('text_bert.encoder.layer.')[1].split('.')[0]) < int(args.knowledge_freeze.split('t')[1]):
                excluded_parameters += [key]
        if 'text_bert' in key:
            module_name = key.split('text_bert')[0] + 'text_bert'
            if module_name not in text_bert_re_init:
                text_bert_re_init.append(module_name)

    for bb in text_bert_re_init:
        exec('model.%s=AutoModel.from_pretrained("prajjwal1/bert-%s")'% (bb, args.bert_size))
    
    for key, value in dict(model.named_parameters()).items():
        if key in excluded_parameters:
            value.requires_grad = False

    optimizer_grouped_parameters = []

    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if key in new_parameters:
                lr = args.new_param_lr
            else:
                lr = args.old_param_lr
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.0, 'pname': key}
                ]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01, 'pname': key}
                ]

    print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    optimizer = AdamW(optimizer_grouped_parameters, lr=base_lr, correct_bias=False)
    warmpu_steps = args.warmup_proportion * num_train_optimization_steps
    warmup_scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmpu_steps, t_total=num_train_optimization_steps
    )

    startIterID = 0
    global_step = 0
    start_epoch = 0

    model.to(device)

    if default_gpu:
        print("***** Running training *****")
        print("  Num Iters: ", task_num_iters)
        print("  Batch size: ", task_batch_size)
        print("  Num steps: %d" % num_train_optimization_steps)

    task_iter_train = {name: None for name in task_ids}
    task_count = {name: 0 for name in task_ids}

    if args.evaluate:
        results = evaluate(
                            args,
                            task_dataloader_val,
                            task_cfg,
                            device,
                            'TASK19',
                            model,
                            task_losses, log_f,
                        )
        import pickle
        pickle.dump(results, open(savePath + '/results.pkl', 'wb'))
        return 0
    epoch_id = 0
    for epochId in range(start_epoch, args.num_train_epochs):
        if epoch_id == args.num_epochs:
            break
        model.train()
        torch.autograd.set_detect_anomaly(True)
        for step in range(median_num_iter):
            iterId = startIterID + step + (epochId * median_num_iter)
            for task_id in task_ids:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                returned_dicts = ForwardModelsTrain(
                        args,
                        task_cfg,
                        device,
                        task_id,
                        task_count,
                        task_iter_train,
                        task_dataloader_train,
                        model,
                        task_losses,
                )
                if step % 10 == 0:
                    string = 'Total: '
                    for key in returned_dicts:
                        if 'loss' in key:
                            string += ' %s: %.4f, ' % (key, returned_dicts[key].item())
                    string = string[:-2] + '\n'
                    log_f.writelines(string)
                    log_f.flush()
                    if step % 20 == 0:
                        print(step, epoch_id, string)
                end.record()
                torch.cuda.synchronize()
                print('Single Iteration Forward Used %.2f ms' % start.elapsed_time(end))

                returned_dicts['loss'].backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                    warmup_scheduler.step()
                    global_step += 1
                end.record()
                torch.cuda.synchronize()
                print('Single Iteration Used %.2f ms' % start.elapsed_time(end))

            # decided whether to evaluate on each tasks.
            for task_id in task_ids:
                if (iterId != 0 and iterId % task_num_iters[task_id] == 0) or (
                        epochId == args.num_train_epochs - 1 and step == median_num_iter - 1
                ):
                    epoch_id += 1
                    results = evaluate(
                        args,
                        task_dataloader_val,
                        task_cfg,
                        device,
                        'TASK19',
                        model,
                        task_losses, log_f,
                    )

                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Only save the model it-self
                    output_model_file = os.path.join(
                        savePath, "pytorch_model.bin"
                    )
                    torch.save(model_to_save.state_dict(), output_model_file)
                    if epoch_id == args.num_epochs:
                        pickle.dump(results, open(savePath + '/results.pkl', 'wb'))


def evaluate(
        args,
        task_dataloader_val,
        task_cfg,
        device,
        task_id,
        model,
        task_losses,
        log_f
):
    from vilbert.vilbert_mavex import VILBertForVLTasks
    from vilbert.task_utils import (
        LoadDatasets,
        LoadLosses,
        ForwardModelsTrain,
        ForwardModelsVal,
    )
    model.eval()
    returned_variables = ['batch_score', 'batch_score_w', 'batch_score_c', 'batch_score_i', 'batch_score_k',
                          'batch_size', 'batch_score_final_pred', 'results', 'batch_score_mavex_k',
                          'batch_score_mavex_w', 'batch_score_mavex_c', 'batch_score_mavex_i']
    for item in returned_variables:
        if 'batch' in item:
            exec(item[6:] + '=0.')
    results = []
    for batch in tqdm(iter(task_dataloader_val[task_id])):
        returned_dicts = ForwardModelsVal(
            args, task_cfg, device, task_id, batch, model, task_losses
        )
        results += returned_dicts['results']
        # for key, val in returned_dicts.items(): exec(key + '=val')
        for item in returned_variables:
            if 'batch' in item:
                exec(item[6:] + '+=returned_dicts["%s"]' % item)

    string = 'VALIDATION: '
    for item in returned_variables:
        if 'score' in item:
            string += '%s: %.4f, ' % (item[6:], eval(item[6:]) / (eval('size') + 0.000001))
    string = string[:-2] + '\n'
    log_f.writelines(string)
    log_f.flush()

    model.train()

    return results


if __name__ == "__main__":
    main()