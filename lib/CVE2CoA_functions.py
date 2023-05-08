import json
import os

import numpy as np
from collections import Counter

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def build_CVE2CoA_devset(CVE_descs, CVE2CoA, CoA2ID, CVE_IDs=None, neg_count=20):
    import random
    random.seed(13)
    coa_pool = list(CoA2ID.keys())
    dev_samples = {}
    for cveid in CVE_IDs:
        cvedesc = CVE_descs.get(cveid)
        cve_coas = list(CVE2CoA[cveid].values())
        neg_coas = list(set(coa_pool).difference(set(cve_coas)))
        if cvedesc:
            if not cvedesc.startswith("** "):
                dev_samples[cveid] = {'query': cvedesc, 'positive': list(), 'negative': list()}
                for coadesc in cve_coas:
                    dev_samples[cveid]['positive'].append(coadesc)
                for negdesc in random.sample(neg_coas, min([neg_count, len(neg_coas)])):
                    dev_samples[cveid]['negative'].append(negdesc)
    return dev_samples


def _addSample(dataset, s2, s1_list, label, reverse=False):
    from sentence_transformers import InputExample
    if not reverse:
        for s1 in s1_list:
            dataset.append(InputExample(texts=[s1, s2], label=label))
    else:
        for s1 in s1_list:
            dataset.append(InputExample(texts=[s2, s1], label=label))
    return dataset


def _get_Negative_TestSamples(text, neg_pool, min_len=4, intersection_rate=0.5):
    if isinstance(text, str):
        text = [text]
    vocab = []
    for t in text:
        vocab += t.split(" ")
    vocab = set(vocab)
    neg_pool = [neg for neg in neg_pool if
                len(neg.split(" ")) >= min_len and len(set(neg.split(" ")).intersection(vocab)) <= round(
                    intersection_rate * len(set(neg.split(" "))))]
    return neg_pool


def get_TrainingDataset(dset, testSet_rate=0.15, neg_rate=20, neg_intersection_rate=0.5):
    from sentence_transformers import InputExample
    from sklearn.model_selection import train_test_split
    import random
    pool_name, pool_desc, pool_vos, pool_trs = [set(), set(), list(), list()]
    for id in dset:
        pool_name.add(dset[id]['name'])
        pool_desc.add(dset[id]['desc'])
        pool_vos += dset[id]['vos']
        pool_trs += dset[id]['trs']
    pool_vos = set(pool_vos)
    pool_trs = set(pool_trs)

    trainSet, testSet = [[], []]
    for techid in dset:
        techname = dset[techid]['name']
        techdesc = dset[techid]['desc']
        vo_train, vo_test = train_test_split(dset[techid]['vos'], test_size=testSet_rate, random_state=13)
        vo_train, vo_test = [list(set(vo_train)), list(set(vo_test))]

        tr_train, tr_test = train_test_split(dset[techid]['trs'], test_size=testSet_rate, random_state=13)
        tr_train, tr_test = [list(set(tr_train)), list(set(tr_test))]

        neg_name_pool = list(pool_name.difference({techname}))
        neg_desc_pool = list(pool_desc.difference({techdesc}))
        neg_vo_pool = list(pool_vos.difference(set(dset[techid]['vos'])))
        neg_vo_pool = _get_Negative_TestSamples(set(dset[techid]['vos']), neg_vo_pool, min_len=4,
                                                intersection_rate=neg_intersection_rate)

        neg_tr_pool = list(pool_trs.difference(set(dset[techid]['trs'])))
        neg_tr_pool = _get_Negative_TestSamples(set(dset[techid]['trs']), neg_tr_pool, min_len=4,
                                                intersection_rate=neg_intersection_rate)

        random.seed(13)
        neg_name_train = random.sample(neg_name_pool, neg_rate)
        neg_desc_train = random.sample(neg_desc_pool, neg_rate)
        neg_vo_train = random.sample(neg_vo_pool, neg_rate)
        neg_tr_train = random.sample(neg_tr_pool, neg_rate)

        random.seed(14)
        neg_name_test = random.sample(list(set(neg_name_pool).difference(set(neg_name_train))), neg_rate)
        neg_desc_test = random.sample(list(set(neg_desc_pool).difference(set(neg_desc_train))), neg_rate)
        neg_vo_test = random.sample(list(set(neg_vo_pool).difference(set(neg_vo_train))), neg_rate)
        neg_tr_test = random.sample(list(set(neg_tr_pool).difference(set(neg_tr_train))), neg_rate)

        # if len(list(set(vo_train).intersection(set(neg_vo_train)))) > 0 or len(list(set(tr_train).intersection(set(neg_tr_train)))):
        #     print("Error in negative sampling in ", techid)
        #     break
        # else:
        #     print("No Error found in negative sampling")
        # add positives _ Train
        trainSet.append(InputExample(texts=[techname, techdesc], label=1.0))  # desc-name
        trainSet = _addSample(trainSet, techdesc, vo_train, label=1.0)  # vo-desc
        trainSet = _addSample(trainSet, techdesc, tr_train, label=1.0)  # tr-desc
        trainSet += [InputExample(texts=[a, b], label=1.0) for idx, a in enumerate(tr_train) for b in
                     tr_train[idx + 1:]]  # tr-tr
        trainSet = _addSample(trainSet, techname, tr_train, label=1.0, reverse=True)  # tr-name

        # add positives _ Test
        testSet = _addSample(testSet, techdesc, vo_test, label=1.0)  # vo-desc
        testSet = _addSample(testSet, techdesc, tr_test, label=1.0)  # tr-desc
        testSet += [InputExample(texts=[a, b], label=1.0) for idx, a in enumerate(tr_test) for b in
                    tr_test[idx + 1:]]  # tr-tr
        testSet = _addSample(testSet, techname, tr_test, label=1.0, reverse=True)  # tr-name

        # add negative _ Train
        c = len(trainSet)
        # trainSet = _addSample(trainSet, techname, neg_name_train, label=0.0)  # name-name
        trainSet = _addSample(trainSet, techdesc, neg_vo_train, label=0.0)  # vo-desc
        trainSet = _addSample(trainSet, techdesc, neg_tr_train, label=0.0)  # tr-desc
        trainSet = _addSample(trainSet, techdesc, neg_desc_train, label=0.0)  # desc-desc
        trainSet = _addSample(trainSet, techname, neg_tr_train, label=0.0, reverse=True)  # tr-name
        print("%d negative sample added" % (len(trainSet) - c))

        # add negative _ Test
        testSet = _addSample(testSet, techdesc, neg_name_test, label=0.0)  # desc-name
        testSet = _addSample(testSet, techdesc, neg_vo_test, label=0.0)  # vo-desc
        testSet = _addSample(testSet, techdesc, neg_tr_test, label=0.0)  # tr-desc
        testSet = _addSample(testSet, techdesc, neg_desc_test, label=0.0)  # desc-desc
        testSet = _addSample(testSet, techname, neg_tr_test, label=0.0, reverse=True)  # tr-name
    random.shuffle(trainSet)
    return trainSet, testSet


def func_import_model(fname):
    import pickle
    return pickle.load(open(fname, 'rb'))


def func_read_texttolist(fname):
    with open(fname) as f:
        mylist = f.read().splitlines()
    return mylist


def func_savejson(DICT, fname):
    with open(fname, 'w', encoding='"iso-8859-1"') as fout:
        json.dump(DICT, fout)


def func_read_json(fname):
    with open(fname, encoding="utf8") as json_file:
        data = json.load(json_file)
    return data


def compute_metrics_all(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def func_merge_listOflist(lst):
    return [j for i in lst for j in i]


def CVE_from_NVD(DIR):
    filenames = []
    NVD_list = []
    for filename in os.listdir(DIR):
        filenames.append(filename)
        NVD_list.append(func_read_json(DIR + filename)['CVE_Items'])
    NVD_list = func_merge_listOflist(NVD_list)
    print("Read NVD from files")
    return NVD_list


def CVE_creatdict(NVD_list, full=False):
    dic = {}
    for cve in NVD_list:
        if full:
            dic[cve['cve']['CVE_data_meta']['ID']] = cve
        else:
            dic[cve['cve']['CVE_data_meta']['ID']] = cve['cve']['description']['description_data'][0]['value']

    return dic

def IR_eval(model, samples, epoch: int = -1, steps: int = -1):
    at_K = 5
    if isinstance(samples, dict):
        samples = list(samples.values())

    all_mrrat5_scores = []
    all_mrrat10_scores = []
    all_mrrat15_scores = []
    all_mrrat20_scores = []

    mrr_scores = {"at5": 0,
                  "at10": 0,
                  "at15": 0,
                  "at20": 0, }

    all_mapat5_scores = []
    all_mapat10_scores = []
    all_mapat15_scores = []
    all_mapat20_scores = []
    map_scores = {"at5": 0,
                  "at10": 0,
                  "at15": 0,
                  "at20": 0, }
    num_queries = 0
    num_positives = []
    num_negatives = []
    for instance in samples:
        query = instance['query']
        positive = list(instance['positive'])
        negative = list(instance['negative'])
        docs = positive + negative
        is_relevant = [True] * len(positive) + [False] * len(negative)

        if len(positive) == 0 or len(negative) == 0:
            continue

        num_queries += 1
        num_positives.append(len(positive))
        num_negatives.append(len(negative))

        model_input = [[query, doc] for doc in docs]
        pred_scores = model.predict(model_input, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
        pred_scores_argsort = np.argsort(-pred_scores)  # Sort in decreasing order

        mrr_score = 0
        for rank, index in enumerate(pred_scores_argsort[0:5]):
            if is_relevant[index]:
                mrr_score = 1 / (rank + 1)
                break
        all_mrrat5_scores.append(mrr_score)

        mrr_score = 0
        for rank, index in enumerate(pred_scores_argsort[0:10]):
            if is_relevant[index]:
                mrr_score = 1 / (rank + 1)
                break
        all_mrrat10_scores.append(mrr_score)

        mrr_score = 0
        for rank, index in enumerate(pred_scores_argsort[0:15]):
            if is_relevant[index]:
                mrr_score = 1 / (rank + 1)
                break
        all_mrrat15_scores.append(mrr_score)

        mrr_score = 0
        for rank, index in enumerate(pred_scores_argsort[0:20]):
            if is_relevant[index]:
                mrr_score = 1 / (rank + 1)
                break
        all_mrrat20_scores.append(mrr_score)

        correct_class_size = Counter(is_relevant)[True]
        map_score = 0

        for rank, index in enumerate(pred_scores_argsort[0:20]):
            if is_relevant[index]:
                map_score += 1

            if rank == 4:
                all_mapat5_scores.append(map_score / min(5, correct_class_size))

            elif rank == 9:
                all_mapat10_scores.append(map_score / min(10, correct_class_size))
            elif rank == 14:
                all_mapat15_scores.append(map_score / min(15, correct_class_size))
            elif rank == 19:
                all_mapat20_scores.append(map_score / min(20, correct_class_size))

    mrr_scores["at5"] = np.mean(all_mrrat5_scores)
    mrr_scores["at10"] = np.mean(all_mrrat10_scores)
    mrr_scores["at15"] = np.mean(all_mrrat15_scores)
    mrr_scores["at20"] = np.mean(all_mrrat20_scores)

    map_scores['at5'] = sum(all_mapat5_scores) / len(samples)
    map_scores['at10'] = sum(all_mapat10_scores) / len(samples)
    map_scores['at15'] = sum(all_mapat15_scores) / len(samples)
    map_scores['at20'] = sum(all_mapat20_scores) / len(samples)

    return mrr_scores, map_scores
