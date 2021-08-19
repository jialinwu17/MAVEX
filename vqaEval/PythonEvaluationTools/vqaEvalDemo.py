# coding: utf-8

import sys
sys.path.insert(0, '/scratch/cluster/jialinwu/mavex/vqaEval/PythonHelperTools/vqaTools')

sys.path.insert(0, '/scratch/cluster/jialinwu/mavex/vqaEval/PythonEvaluationTools')
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import matplotlib.pyplot as plt
import json
import random
import os, pickle

def evaluate_vqa(pickle_results):
    annFile = '/scratch/cluster/jialinwu/mavex/data/okvqa_new/cache/tmp/mscoco_val2014_annotations.json'
    quesFile = '/scratch/cluster/jialinwu/mavex/data/okvqa_new/cache/tmp/OpenEnded_mscoco_val2014_questions.json'
    label2ans = pickle.load(open('/scratch/cluster/jialinwu/mavex/data/okvqa_new/cache/trainval_label2ans_729.pkl', 'rb'))
    resFile = []
    qid2gt_answer = pickle.load(
        open("/scratch/cluster/jialinwu/mavex/data/okvqa_new/cache/tmp/val_qid2answers_keep_0.pkl", 'rb'))
    qids = sorted(list(qid2gt_answer.keys()))
    ct = 0
    for res in pickle_results:
        if 'question_id' in res:
            qid = res['question_id']
        else:
            qid = qids[ct]
            ct += 1
        if 'str_pred' in res:
            resFile.append({'answer': res['str_pred'], 'question_id':qid})
        else:
            resFile.append({'answer': label2ans[res['final_pred'].argmax()], 'question_id':qid})

    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    return vqaEval.accuracy['overall']

def evaluate_good_or_vqa(pickle_results):
    annFile = '/scratch/cluster/jialinwu/mavex/data/okvqa_new/cache/tmp/mscoco_val2014_annotations.json'
    quesFile = '/scratch/cluster/jialinwu/mavex/data/okvqa_new/cache/tmp/OpenEnded_mscoco_val2014_questions.json'
    label2ans = pickle.load(open('/scratch/cluster/jialinwu/mavex/data/okvqa_new/cache/trainval_label2ans_729.pkl', 'rb'))
    ques = json.load(
        open('/scratch/cluster/jialinwu/mavex/data/okvqa_new/cache/tmp/OpenEnded_mscoco_val2014_questions.json'))['questions']
    qid2question = {}
    for q in ques:
        qid2question[q['question_id']] = q['question'].lower()
    resFile = []

    qids = sorted(list(qid2question.keys()))
    ct = 0
    for res in pickle_results:
        if 'question_id' in res:
            qid = res['question_id']
        else:
            qid = qids[ct]
            ct += 1

        question = qid2question[qid]
        pred_argsort = res['final_pred'].argsort()[::-1]
        answer = label2ans[pred_argsort[0]]
        if ' or ' in question:
            for i in range(1000):
                tmp_ans = label2ans[pred_argsort[i]]
                if tmp_ans in question:
                    if len(tmp_ans) < 2:
                        continue
                    answer = tmp_ans
                    break
                else:
                    words = tmp_ans.split()

                    if len(words):
                        a = 1
                        for word in words:
                            if word not in question:
                                a = 0
                                break
                        if a:
                            answer = tmp_ans
                            print(answer)

        resFile.append({'answer':answer, 'question_id': qid})

    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    return vqaEval.accuracy['overall']

'''
import pickle
pickle_results = pickle.load(open('/scratch/cluster/jialinwu/mavex/save/OKVQAMAVExBREAKSEP_bert_base_6layer_6conect-eval1/results.pkl', 'rb'))
print(evaluate_good_or_vqa(pickle_results))
'''