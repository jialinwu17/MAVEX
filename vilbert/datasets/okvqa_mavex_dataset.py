# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import _pickle as cPickle
import pickle
import logging
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from ._image_features_reader import ImageFeaturesH5Reader
import copy
from transformers import AutoTokenizer, AutoModel


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from .feat_utils import read_features
from .concept_utils import concept_to_sentence

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(question, answer):
    # answer.pop("image_id")
    # answer.pop("question_id")
    entry = {
        "question_id": question["question_id"],
        "image_id": int(question["question_id"] // 10),
        "question": question["question"],
        "answer": answer,
    }
    return entry


def _load_dataset(dataroot, name, clean_datasets):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'
    """

    question_path = os.path.join(
        dataroot, "OpenEnded_mscoco_%s2014_questions.json" % name
    )
    questions = sorted(
        json.load(open(question_path))["questions"], key=lambda x: x["question_id"]
    )

    answer_path = os.path.join('data/okvqa_new', "cache", "%s_target_729.pkl" % (name))
    answers = cPickle.load(open(answer_path, "rb"))
    answers = sorted(answers, key=lambda x: x["question_id"])
    assert_eq(len(questions), len(answers))
    entries = []
    id = 0
    for question in questions:
        answer = answers[id]
        id += 1
        assert_eq(question["question_id"], answer["question_id"])
        # assert_eq(question["image_id"], answer["image_id"])
        entries.append(_create_entry(question, answer))

    return entries


class VQAClassificationDataset(Dataset):
    def __init__(
            self,
            task,
            dataroot,
            annotations_jsonpath,
            split,
            image_features_reader,
            gt_image_features_reader,
            tokenizer,
            bert_model,
            clean_datasets=None,
            padding_index=0,
            max_seq_length=16,
            max_region_num=101,
            args=None,
    ):
        super().__init__()
        self.split = split
        ans2label_path = os.path.join('data/okvqa_new', "cache",  "trainval_ans2label_729.pkl")
        label2ans_path = os.path.join('data/okvqa_new', "cache", "trainval_label2ans_729.pkl" )
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self.separate = 0
        
        self.num_wiki_sentences = args.num_wiki_sentences
        self.num_concepts = args.num_concepts
        self.wiki_seq_length = 20
        self.concept_seq_length = 14

        if split == 'test':
            new_split = 'val'
        else:
            new_split = 'train'
        self.entries = _load_dataset(dataroot, new_split, clean_datasets)
        self.image_feats_hf = h5py.File('h5py_accumulate/image_%s.hdf5' % new_split, 'r')
        self.image_feats = self.image_feats_hf.get('features')
        self.image_qid2ind = cPickle.load(open('h5py_accumulate/image_%s_qid_ans2idx.pkl' % new_split, 'rb'))

        self.use_search_word = args.use_search_word
        self.qid2results = pickle.load(open('qid2answer_candidates_trainval.pkl', 'rb'))
        self.qid2results.update(pickle.load(open('qid2answer_candidates_test.pkl', 'rb')))
        self.qid2knowledge = cPickle.load(open('qid2knowledge_cleaned_mavex.pkl', 'rb'))
        self.small_bert_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
        self.id2path = cPickle.load(open('data/coco_iid2feat_path.pkl', 'rb'))
        self.tokenize(max_seq_length)
        self.tensorize()
        self.targets = pickle.load(open('data/okvqa_new/cache/%s_target_729.pkl' % new_split, 'rb'))
        self.qid2gt_answer = {}
        for val_target in self.targets:
            question_id = val_target['question_id']
            labels = val_target['labels']
            scores = val_target['scores']
            self.qid2gt_answer[question_id] = {}
            for i in range(len(labels)):
                self.qid2gt_answer[question_id][self.label2ans[labels[i]]] = scores[i]

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["question"])
            tokens = tokens[: max_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tokenize_sentence(self, sentence, sw='', max_length=16):
        if len(sw):
            # tokens = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(sentence) + ['[SEP]'] + self._tokenizer.tokenize(sw))
            tokens1 = self.small_bert_tokenizer.convert_tokens_to_ids(self.small_bert_tokenizer.tokenize(sentence))
            segment_ids1 = [0] * len(tokens1)
            tokens1 = tokens1[: max_length - 3 - 4]
            segment_ids1 = segment_ids1[: max_length - 3 - 4]

            tokens2 = self.small_bert_tokenizer.convert_tokens_to_ids(self.small_bert_tokenizer.tokenize(sw))
            segment_ids2 = [1] * len(tokens2)
            tokens2 = tokens2[:  4]
            segment_ids2 = segment_ids2[: 4]
            # SEP is 102 CLS 101 PAD 0
            tokens = [101] + tokens1 + [102] + tokens2 + [102]
            segment_ids = [0] + segment_ids1 + [0] + segment_ids2 + [1]
        else:
            tokens = self.small_bert_tokenizer.convert_tokens_to_ids(self.small_bert_tokenizer.tokenize(sentence))
            tokens = [101] + tokens[: max_length - 2] + [102]
            segment_ids = [0] * len(tokens)

        input_mask = [1] * len(tokens)

        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [0] * (max_length - len(tokens))
            tokens = tokens + padding
            input_mask += padding
            segment_ids += padding

        return tokens, input_mask, segment_ids

    def tensorize(self):

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            # if "test" not in self.split:
            if True:
                answer = entry["answer"]
                labels = np.array(answer["labels"])
                scores = np.array(answer["scores"], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry["answer"]["labels"] = labels
                    entry["answer"]["scores"] = scores
                else:
                    entry["answer"]["labels"] = None
                    entry["answer"]["scores"] = None

    def __getitem__(self, index):
        #  mode 1 no share | share features = zero(1, 768)
        #  mode 2 only share score 0 features
        #  mode 3 share all features

        entry = self.entries[index]
        image_id = entry["image_id"]
        question_id = entry["question_id"]
        features, num_boxes, boxes, _ = read_features(self.id2path[image_id])
        # features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
        target = torch.zeros(self.num_labels)
        wiki_tokens, wiki_input_masks, wiki_segment_ids = self.gather_wiki_features(question_id, self.wiki_seq_length)
        concept_tokens, concept_input_masks, concept_segment_ids = self.gather_concept_features(question_id, self.concept_seq_length)
        # print('concepting')

        verif_answers = []
        verif_labels = []
        verif_inds = []
        ctt = 0
        for pred_answer in self.qid2results[question_id][:5]:
            verif_answers.append(pred_answer)
            verif_inds.append(self.ans2label[pred_answer])
            if pred_answer in self.qid2gt_answer[question_id]:
                verif_labels.append(self.qid2gt_answer[question_id][pred_answer])
                ctt += 1
            else:
                verif_labels.append(0.)
                ctt += 1
        verif_labels = torch.from_numpy(np.array(verif_labels)).float()
        verif_inds = torch.from_numpy(np.array(verif_inds)).float()
        google_images = []

        for pred_answer in verif_answers:
            google_images_per_answer = np.zeros((5, 2048))
            if question_id in self.image_qid2ind and pred_answer in self.image_qid2ind[question_id]:
                if self.image_qid2ind[question_id][pred_answer] >= 0:
                    google_images_per_answer = np.array(self.image_feats[self.image_qid2ind[question_id][pred_answer]])
            google_images.append(google_images_per_answer)
        google_images = torch.from_numpy(np.stack(google_images, 0)).float()

        answer = entry["answer"]
        labels = answer["labels"]
        scores = answer["scores"]
        if labels is not None:
            target.scatter_(0, labels, scores)

        return (
            features,
            spatials,
            image_mask,
            question,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            question_id,
            wiki_tokens,
            concept_tokens,
            google_images,
            verif_labels,
            verif_inds,
        )

    def __len__(self):
        return len(self.entries)

    def gather_wiki_features(self, question_id, max_length=22):
        num_wiki_sentences_total = self.num_wiki_sentences
        wiki_tokens = []
        wiki_input_masks = []
        wiki_segment_ids = []

        if self.separate:
            wiki_sentences = self.qid2knowledge[question_id]['separate_wiki_sentences'][:num_wiki_sentences_total]
        else:
            wiki_sentences = self.qid2knowledge[question_id]['non_separate_wiki_sentences'][:num_wiki_sentences_total]

        for wiki in wiki_sentences:
            assert len(wiki.split(' [SEP] ')) == 2
            search_word = wiki.split(' [SEP] ')[0]
            sent = wiki.split(' [SEP] ')[1]

            token, input_mask, segment_id = self.tokenize_sentence(sent,
                                                                       search_word if self.use_search_word else '',
                                                                       max_length)
            wiki_tokens.append(token)
            wiki_input_masks.append(input_mask)
            wiki_segment_ids.append(segment_id)
        current_num_sentences = len(wiki_tokens)
        for i in range(current_num_sentences, num_wiki_sentences_total):
            wiki_tokens.append([0] * max_length)
            wiki_input_masks.append([0] * max_length)
            wiki_segment_ids.append([0] * max_length)

        wiki_tokens = torch.from_numpy(np.array(wiki_tokens).astype('int'))
        wiki_input_masks = torch.from_numpy(np.array(wiki_input_masks).astype('int'))
        wiki_segment_ids = torch.from_numpy(np.array(wiki_segment_ids).astype('int'))
        return wiki_tokens, wiki_input_masks, wiki_segment_ids

    def gather_concept_features(self, question_id, max_length=18):
        num_concept_sentences_total = self.num_concepts
        concept_tokens = []
        concept_input_masks = []
        concept_segment_ids = []
        if self.separate:
            concept_strs = self.qid2knowledge[question_id]['separate_concept_strs'][:num_concept_sentences_total]
        else:
            concept_strs = self.qid2knowledge[question_id]['non_separate_concept_strs'][:num_concept_sentences_total]

        for concept_str in concept_strs:
            concept_sent = concept_to_sentence(concept_str.split(' [SEP] ')[1].split('<SEP>'))
            search_word = concept_str.split(' [SEP] ')[0]
            token, input_mask, segment_id = self.tokenize_sentence(concept_sent,
                                                                       search_word if self.use_search_word else '',
                                                                       max_length)
            concept_tokens.append(token)
            concept_input_masks.append(input_mask)
            concept_segment_ids.append(segment_id)
        current_num_sentences = len(concept_tokens)
        for i in range(current_num_sentences, num_concept_sentences_total):
            concept_tokens.append([0] *max_length )
            concept_input_masks.append([0] *max_length )
            concept_segment_ids.append([0] *max_length )
        concept_tokens = torch.from_numpy(np.array(concept_tokens).astype('int'))
        concept_input_masks = torch.from_numpy(np.array(concept_input_masks).astype('int'))
        concept_segment_ids = torch.from_numpy(np.array(concept_segment_ids).astype('int'))
        return concept_tokens, concept_input_masks, concept_segment_ids


