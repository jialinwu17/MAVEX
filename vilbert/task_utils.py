# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from io import open
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.tokenization_bert import BertTokenizer
import pdb
import numpy as np

logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
}


def mis_match_loss_max(pred, verif_labels, mave_mask, args):
    bce = nn.BCELoss(reduction='none')
    num_answer = pred.size(1)
    num_correct_answers = (verif_labels > 0.).long().sum(1) # batch
    losses2 = bce(pred[:, np.arange(num_answer), np.arange(num_answer)], verif_labels.float()).sum(1).mean() * args.mml2

    diag_mask = np.ones((pred.size(0), 5, 5))
    for i in range(pred.size(0)):
        diag_mask[i, :num_correct_answers[i], :num_correct_answers[i]] = 0.
    diag_mask = torch.from_numpy(diag_mask).float().cuda()
    diag_mask[:, np.arange(5), np.arange(5)] = 0.
    diag_mask = diag_mask *  mave_mask.view(pred.size(0), 1, -1)

    diag_loss = bce(pred,  torch.zeros((verif_labels.size(0), num_answer, num_answer)).float().cuda())
    diag_loss = diag_loss * diag_mask
    losses1 = diag_loss.max(1)[0]  # batch 5
    losses1 = (losses1.sum(1) / ((losses1 > 0.).sum(1)+ 0.000001)).mean()* args.mml0

    losses0 = diag_loss.max(2)[0]  # batch 5
    losses0 = (losses0.sum(1) / ((losses0 > 0.).sum(1) + 0.000001)).mean()* args.mml0

    losses = losses2 + losses1 + losses0

    return losses


def ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch if torch.is_tensor(t) )

    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id, \
        wiki_tokens, concept_tokens, google_images, verif_labels, verif_inds = batch

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
    vil_pred, knowledge_pred, wiki_pred, concept_pred, image_pred, knowledge_mavex_pred, wiki_mavex_pred, concept_mavex_pred, \
    image_mavex_pred, knowledge_mavex_mask, wiki_mavex_mask, concept_mavex_mask, image_mavex_mask = model(
        question,
        features,
        spatials,
        segment_ids,
        input_mask,
        wiki_tokens=wiki_tokens,  # batch, 4(search words), 10 (sents) 18 (tokens)
        concept_tokens=concept_tokens,
        google_images=google_images,  # batch, 4(search words), 10 (sents) 18 (tokens)
        question_ids = question_id,
        image_attention_mask=image_mask,
        co_attention_mask=co_attention_mask,
        task_ids=task_tokens,
        verif_inds=verif_inds,
    )
    sigmoid = torch.nn.Sigmoid()
    num_answers = wiki_mavex_pred.size(1)
    final_pred = torch.stack([vil_pred, knowledge_pred], 1).max(1)[0]  # batch, num_search, 5020
    pred_prob = sigmoid(torch.gather(final_pred, 1, verif_inds.long()))

    vil_pred_np = vil_pred.detach().cpu().numpy()
    knowledge_prediction_np = knowledge_pred.detach().cpu().numpy()
    final_pred_np = final_pred.detach().cpu().numpy()
    wiki_pred_np = wiki_pred.detach().cpu().numpy()
    concept_pred_np = concept_pred.detach().cpu().numpy()
    image_pred_np = image_pred.detach().cpu().numpy()
    wiki_mavex_pred_np = wiki_mavex_pred.detach().cpu().numpy()
    concept_mavex_pred_np = concept_mavex_pred.detach().cpu().numpy()
    image_mavex_pred_np = image_mavex_pred.detach().cpu().numpy()
    verif_inds_np = verif_inds.detach().cpu().numpy()
    wiki_mavex_mask_np = wiki_mavex_mask.detach().cpu().numpy()
    concept_mavex_mask_np = concept_mavex_mask.detach().cpu().numpy()
    image_mavex_mask_np = image_mavex_mask.detach().cpu().numpy()
    knowledge_mavex_mask_np = knowledge_mavex_mask.detach().cpu().numpy()
    knowledge_mavex_pred_np = knowledge_mavex_pred.detach().cpu().numpy()

    results = []
    batch_size = len(question_id)
    for i in range(batch_size):
        results.append({'vil_pred': vil_pred_np[i], 'knowledge_pred': knowledge_prediction_np[i],
                        'question_id': int(question_id[i]),  'wiki_pred': wiki_pred_np[i],
                        'final_pred': final_pred_np[i], 'concept_pred': concept_pred_np[i],
                        'image_pred': image_pred_np[i], 'wiki_mavex_pred': wiki_mavex_pred_np[i],
                        'concept_mavex_pred': concept_mavex_pred_np[i], 'verif_inds' : verif_inds_np[i],
                        'image_mavex_pred': image_mavex_pred_np[i], 'wiki_mavex_mask': wiki_mavex_mask_np[i],
                        'concept_mavex_mask': concept_mavex_mask_np[i],
                        'image_mavex_mask' : image_mavex_mask_np[i],
                        'knowledge_mavex_pred':knowledge_mavex_pred_np[i],
                        'knowledge_mavex_mask' : knowledge_mavex_mask_np[i]})

    batch_score = compute_score_with_logits(vil_pred, target).sum()
    batch_score_final_pred = compute_score_with_logits(final_pred, target).sum()
    batch_score_w = compute_score_with_logits(wiki_pred, target).sum()
    batch_score_c = compute_score_with_logits(concept_pred, target).sum()
    batch_score_i = compute_score_with_logits(image_pred, target).sum()
    batch_score_k = compute_score_with_logits(knowledge_pred, target).sum()

    batch_score_mavex_k = compute_score_with_logits(knowledge_mavex_pred[:, np.arange(num_answers), np.arange(num_answers)] * pred_prob, verif_labels).sum()
    batch_score_mavex_w = compute_score_with_logits(wiki_mavex_pred[:, np.arange(num_answers), np.arange(num_answers)]* pred_prob, verif_labels).sum()
    batch_score_mavex_c = compute_score_with_logits(concept_mavex_pred[:, np.arange(num_answers), np.arange(num_answers)]* pred_prob, verif_labels).sum()
    batch_score_mavex_i = compute_score_with_logits(image_mavex_pred[:, np.arange(num_answers), np.arange(num_answers)]* pred_prob, verif_labels).sum()

    returned_variables = [ 'batch_score', 'batch_score_w', 'batch_score_c', 'batch_score_i', 'batch_score_k',
                           'batch_size', 'batch_score_final_pred', 'results', 'batch_score_mavex_k',
                           'batch_score_mavex_w', 'batch_score_mavex_c','batch_score_mavex_i']
    return_dict = {}
    for name in returned_variables:
        tmp = eval(name)
        if torch.is_tensor(tmp):
            tmp = tmp.detach().cpu().numpy()
        return_dict[name] = tmp
    return return_dict


def ForwardModelsTrain(
        args,
        task_cfg,
        device,
        task_id,
        task_count,
        task_iter_train,
        task_dataloader_train,
        model,
        task_losses,
):
    # given the current task, decided whether to forward the model and forward with specific loss.

    # reset the task iteration when needed.
    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])

    task_count[task_id] += 1
    batch = task_iter_train[task_id].next()
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch if torch.is_tensor(t))

    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id, \
        wiki_tokens, concept_tokens, google_images, verif_labels, verif_inds = batch

    task_tokens = question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
    bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction="none")

    vil_pred, knowledge_pred, wiki_pred, concept_pred, image_pred, knowledge_mavex_pred, wiki_mavex_pred, concept_mavex_pred, \
    image_mavex_pred, knowledge_mavex_mask, wiki_mavex_mask, concept_mavex_mask, image_mavex_mask = model(
        question,
        features,
        spatials,
        segment_ids,
        input_mask,
        wiki_tokens=wiki_tokens,  # batch, 4(search words), 10 (sents) 18 (tokens)
        concept_tokens=concept_tokens,
        google_images=google_images,  # batch, 4(search words), 10 (sents) 18 (tokens)
        question_ids = question_id,
        image_attention_mask=image_mask,
        co_attention_mask=co_attention_mask,
        task_ids=task_tokens,
        verif_inds=verif_inds,
    )
    loss_wiki_pred = bce_with_logits_loss(wiki_pred, target).sum(1).mean()
    loss_concept_pred = bce_with_logits_loss(concept_pred, target).sum(1).mean()
    loss_image_pred = bce_with_logits_loss(image_pred, target).sum(1).mean()
    loss_knowledge_pred = bce_with_logits_loss(knowledge_pred, target).sum(1).mean()
    loss_vqa = bce_with_logits_loss(vil_pred, target).sum(1).mean() * args.loss_vqa

    loss_knowledge_pred = loss_concept_pred + loss_wiki_pred + loss_knowledge_pred + loss_image_pred
    loss_knowledge_pred *= args.loss_knowledge_pred

    loss_wiki_diag = mis_match_loss_max(wiki_mavex_pred, verif_labels, wiki_mavex_mask, args)
    loss_concept_diag = mis_match_loss_max(concept_mavex_pred, verif_labels, concept_mavex_mask, args)
    loss_image_diag = mis_match_loss_max(image_mavex_pred, verif_labels, image_mavex_mask, args)
    loss_mavex =  loss_wiki_diag + loss_concept_diag + loss_image_diag

    loss = loss_vqa + loss_knowledge_pred + loss_mavex
    return_dict = {}
    returned_variables = ['loss', 'loss_vqa', 'loss_knowledge_pred', 'loss_mavex']
    for name in returned_variables:
        return_dict[name] = eval(name)
    return return_dict


def LoadLosses(args, task_cfg, task_ids):
    losses = {}
    task_types = []
    num_labels = 0
    for i, task_id in enumerate(task_ids):
        task = "TASK" + task_id 
        model_type = task_cfg[task]["type"]
        if model_type not in task_types:
            task_types.append(model_type)
        losses[task] = LossMap[task_cfg[task]["loss"]]

    return losses


def LoadDatasets(args, task_cfg, ids, split="trainval"):
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )
    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}
    from vilbert.datasets.okvqa_mavex_dataset import VQAClassificationDataset

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_name = task_cfg[task]["name"]
        task_ids.append(task)
        batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps
        num_workers = args.num_workers
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())
            num_workers = int(num_workers / dist.get_world_size())

        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        task_datasets_train[task] = None
        if "train" in split:
            task_datasets_train[task] = VQAClassificationDataset(
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
                split=task_cfg[task]["train_split"],
                image_features_reader={},
                gt_image_features_reader= {},
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"], args=args,
            )

        task_datasets_val[task] = None
        if "val" in split:
            task_datasets_val[task] = VQAClassificationDataset(
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
                split=task_cfg[task]["val_split"],
                image_features_reader={},
                gt_image_features_reader= {},
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"], args=args,
            )

        task_num_iters[task] = 0
        task_batch_size[task] = 0
        if "train" in split:
            if args.local_rank == -1:
                train_sampler = RandomSampler(task_datasets_train[task])
            else:
                # TODO: check if this works with current data generator from disk that relies on next(file)
                # (it doesn't return item back by index)
                train_sampler = DistributedSampler(task_datasets_train[task])

            task_dataloader_train[task] = DataLoader(
                task_datasets_train[task],
                sampler=train_sampler,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )

            task_num_iters[task] = len(task_dataloader_train[task])
            task_batch_size[task] = batch_size

        if "val" in split:
            task_dataloader_val[task] = DataLoader(
                task_datasets_val[task],
                shuffle=False,
                batch_size=16,
                num_workers=2,
                pin_memory=True,
            )

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_train,
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
    )


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores