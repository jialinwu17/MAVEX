import json, pickle, math
import numpy as np
import sys, copy
from vqaEval.PythonEvaluationTools.vqaEvalDemo import evaluate_vqa, evaluate_good_or_vqa


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


results_file = ''
results = pickle.load(open(results_file, 'rb'))
annotations = json.load(open('data/okvqa_new/mscoco_val2014_annotations_729.json', 'r'))
question_types = annotations['question_types']
val_targets = pickle.load(open("data/okvqa_new/cache/val_target_729.pkl", 'rb'))

answer2label = pickle.load(open('data/okvqa_new/cache/trainval_ans2label_729.pkl', 'rb'))
label2ans = pickle.load(open('data/okvqa_new/cache/trainval_label2ans_729.pkl', 'rb'))
qid2q = pickle.load(open('../parse_segments/qid2q.pkl', 'rb'))
qid2gt_answer = {}
for val_target in val_targets:
    question_id = val_target['question_id']
    labels = val_target['labels']
    scores = val_target['scores']
    qid2gt_answer[question_id] = {}
    for i in range(len(labels)):
        qid2gt_answer[question_id][label2ans[labels[i]].lower()] = scores[i]

annos = {}
for annotation in annotations['annotations']:
    annos[int(annotation['question_id'])] = question_types[annotation['question_type']]

detailed_results = {}
for question_type_key in question_types:
    detailed_results[question_types[question_type_key]] = []
detailed_results['Overall'] = []


def mavex_reweight_case_1(vqa_pred, verif_pred, verif_inds, k=5, qid=305):
    # consider top 5 answers in vqa_pred, correct the top-1 prediction by joint prob if there are answer candidates in verif_inds.
    # vqa_pred (5020) verif_pred (5, 5) verif_inds (5), no diag is applied
    argsort_vqa_pred = np.argsort(vqa_pred)[::-1]
    sort_vqa_pred = np.sort(vqa_pred)[::-1]
    pred_answer = argsort_vqa_pred[0]
    joint_prob = np.zeros((k))
    verif_inds = verif_inds.tolist()
    diag_prob = verif_pred[np.arange(5), np.arange(5)]
    if argsort_vqa_pred[0] not in verif_inds:
        joint_pred_ind = pred_answer
    else:
        for ii in range(k):
            if argsort_vqa_pred[ii] in verif_inds:
                joint_prob[ii] = sigmoid(sort_vqa_pred[ii]) * diag_prob[verif_inds.index(argsort_vqa_pred[ii])]
        joint_pred_ind = argsort_vqa_pred[np.argsort(joint_prob)[-1]]
    return label2ans[joint_pred_ind]


def mavex_reweight_case_2(vqa_pred, verif_pred, verif_inds, k=5, qid=305):
    # consider top 5 answers in vqa_pred, correct the top-1 prediction by joint prob if there are answer candidates in verif_inds.
    # vqa_pred (5020) verif_pred (5, 5) verif_inds (5), diag is applied
    argsort_vqa_pred = np.argsort(vqa_pred)[::-1]
    sort_vqa_pred = np.sort(vqa_pred)[::-1]
    pred_answer = argsort_vqa_pred[0]
    joint_prob = np.zeros(k)
    verif_inds = verif_inds.tolist()
    diag_prob = verif_pred[np.arange(5), np.arange(5)]
    if argsort_vqa_pred[0] not in verif_inds:
        joint_pred_ind = pred_answer
    else:
        for ii in range(k):
            if argsort_vqa_pred[ii] in verif_inds:
                ind = verif_inds.index(argsort_vqa_pred[ii])
                if ((verif_pred[ind, ind] == verif_pred[ind].max()) and (verif_pred[ind, ind] == verif_pred[:, ind].max())) or ii == 0:
                    joint_prob[ii] = sigmoid(sort_vqa_pred[ii]) * diag_prob[ind]
        joint_pred_ind = argsort_vqa_pred[np.argsort(joint_prob)[-1]]
    return  label2ans[joint_pred_ind]


def mavex_reweight_case_3(vqa_pred, verif_pred, verif_inds, k=5, qid = 305):
    argsort_vqa_pred = np.argsort(vqa_pred)[::-1]
    vqa_pred_ = copy.copy(vqa_pred)
    joint_prob = np.zeros((k))
    verif_inds = verif_inds.tolist()
    diag_prob = verif_pred[np.arange(5), np.arange(5)]

    question = qid2q[qid]
    rm_ct = 0
    ttt = 1000
    if ' or ' in question:
        for i in range(ttt):
            remove = 1
            tmp_ans = label2ans[argsort_vqa_pred[i]]
            if tmp_ans in question:
                if len(tmp_ans) < 2:
                    remove = 1
                else:
                    remove = 0
            else:
                words = tmp_ans.split()
                if len(words):
                    a = 1
                    for word in words:
                        if word not in question:
                            a = 0
                            break
                    if a:
                        remove = 0
            if remove:
                rm_ct += 1
                vqa_pred[argsort_vqa_pred[i]] = -11111110.

    if rm_ct == ttt:
        vqa_pred = vqa_pred_

    argsort_vqa_pred = np.argsort(vqa_pred)[::-1]

    sort_vqa_pred = np.sort(vqa_pred)[::-1]

    if argsort_vqa_pred[0] not in verif_inds:
        joint_pred_ind = argsort_vqa_pred[0]
    else:
        for i in range(k):
            if argsort_vqa_pred[i] in verif_inds:
                ind = verif_inds.index(argsort_vqa_pred[i])
                joint_prob[i] = sigmoid(sort_vqa_pred[i]) * diag_prob[ind]
        joint_pred_ind = argsort_vqa_pred[np.argsort(joint_prob)[-1]]

    return label2ans[joint_pred_ind]


print(evaluate_vqa(copy.deepcopy(results)))
print(evaluate_good_or_vqa(copy.deepcopy(results)))

reses = []
for i in range(len(results)):
    score = 0.

    result = results[i]
    qid = int(result['question_id'])
    verif_inds = result['verif_inds']
    verif_pred = np.stack([result['wiki_mavex_pred'], result['image_mavex_pred'], result['concept_mavex_pred']]).max(0)
    result = mavex_reweight_case_2(results[i]['final_pred'], verif_pred, verif_inds, qid=qid)
    if result in qid2gt_answer[qid]:
        score = qid2gt_answer[qid][result]
    detailed_results[annos[qid]].append(score)
    detailed_results['Overall'].append(score)
    reses.append({'question_id': qid, 'str_pred': result})

print(evaluate_vqa(reses))

for key in detailed_results:
    print(key, ':', sum(detailed_results[key]) / len(detailed_results[key]))




