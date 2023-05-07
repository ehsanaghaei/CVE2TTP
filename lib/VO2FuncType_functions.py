import json
import logging
import math
import random
from collections import defaultdict

import pandas as pd
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator, CEBinaryAccuracyEvaluator, \
    CEBinaryClassificationEvaluator, CERerankingEvaluator, CESoftmaxAccuracyEvaluator

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, accuracy_score
from transformers import RobertaTokenizerFast

logging.warning('Started')
import os
random.seed(14)

# ----------------------------------------------- Functions

def CVE_creatdict(NVD_list, full=False):
    dic = {}
    for cve in NVD_list:
        if full:
            dic[cve['cve']['CVE_data_meta']['ID']] = cve
        else:
            dic[cve['cve']['CVE_data_meta']['ID']] = cve['cve']['description']['description_data'][0]['value']

    return dic


def func_read_json(fname):
    import json
    with open(fname, encoding="utf8") as json_file:
        data = json.load(json_file)
    return data


def func_merge_listOflist(lst):
    return [j for i in lst for j in i]

def merge_dictionaries(dict_list):
    op_dict = {}
    for d in dict_list:
        for k in d:
            if k not in op_dict:
                op_dict[k] = {}
            for k2 in d[k]:
                op_dict[k][k2] = d[k][k2]
    return op_dict

def make_multihot(lbl_list, num_classes, to_list=True, device='cuda'):
    import numpy as np
    import torch
    from torch import nn
    if type(lbl_list) != list:
        lbl_list = [lbl_list]
    labels = torch.LongTensor(lbl_list)
    # labels = torch.DoubleTensor(lbl_list)
    # torch.DoubleTensor

    y_onehot = nn.functional.one_hot(labels, num_classes=num_classes)
    # y_onehot = y_onehot.sum(dim=0).to(device).float()
    y_onehot = y_onehot.sum(dim=0).float()
    if to_list:
        y_onehot = y_onehot.numpy()
    return y_onehot


def CVE_from_NVD(DIR):
    filenames = []
    NVD_list = []
    for filename in os.listdir(DIR):
        filenames.append(filename)
        NVD_list.append(func_read_json(DIR + filename)['CVE_Items'])
    NVD_list = func_merge_listOflist(NVD_list)
    print("Read NVD from files")
    return NVD_list


def merge_SVO(op2, BL_verb=None, BL_CVEids=[], lower=True):
    if BL_verb is None:
        BL_verb = []
    from collections import defaultdict
    sents = defaultdict(list)
    for cve_id in op2:
        if cve_id not in BL_CVEids:
            for svo in op2[cve_id]:
                if svo[1].startswith("V:") and svo[1].split("V: ")[1] not in BL_verb:
                    sent = " ".join([s.split(': ')[1] for s in svo])
                    sent = sent.replace(" - ", "-")
                    if lower:
                        sent = sent.lower()
                    sents[cve_id].append(sent)
    return sents


def parse_VOs(dic, arg, word, ex='ADV:'):
    from collections import defaultdict
    op = defaultdict(list)

    for id in dic:
        for rec in dic[id]:
            for r in rec:
                if type(word) != list:
                    if arg in r and word.lower() in r.lower() and ex not in r.lower():
                        op[id].append(rec)
                else:
                    if arg in r and len([w for w in word if w.lower() in r.lower() and ex not in r]) > 0:
                        op[id].append(rec)
    return op


def sort_dict(dict):
    sorted_dict = {}
    sorted_keys = sorted(dict, key=dict.get, reverse=True)
    for w in sorted_keys:
        sorted_dict[w] = dict[w]
    return sorted_dict


def get_args(source, arg):
    from collections import Counter
    op = []
    for id in source:
        lst = source[id]
        for rec in lst:
            for sub_rec in rec:
                if arg in sub_rec:
                    op.append(sub_rec.split(arg)[1])
    counter = Counter(op)
    return sort_dict(counter)


def dataset_reverse(dataset):
    dset_rev = defaultdict(list)
    dsetType_rev = defaultdict(list)
    for typ in dataset:
        for cveid in dataset[typ]:
            svos = dataset[typ][cveid]
            for svo in svos:
                dset_rev[svo].append(cveid)
                dset_rev[svo] = list(set(dset_rev[svo]))
                dsetType_rev[svo].append(typ)
                dsetType_rev[svo] = list(set(dsetType_rev[svo]))
    return dset_rev, dsetType_rev

def clean_manual_clean(dataset_manual_clean):
    op = {}
    for typ in dataset_manual_clean:
        op[typ] = {}
        for cid in dataset_manual_clean[typ]['pos']:
            if dataset_manual_clean[typ]['pos'][cid] and dataset_manual_clean[typ]['pos'][cid] !=[""] and dataset_manual_clean[typ]['pos'][cid] !=[]:
                lst = dataset_manual_clean[typ]['pos'][cid]
                lst = [r for r in lst if r]
                op[typ][cid] = lst
        dataset_manual_clean[typ]['pos'] = op[typ]

    return dataset_manual_clean
class Create_TrainDataset:
    def __init__(self, Config):
        self.Config = Config
        self.dataset = self.sanitize_dataset(func_read_json(os.path.join(Config.path_VO_data, 'CVE_VOs4.json')))
        self.dataset_rev, datasetType_rev = dataset_reverse(self.dataset)
        self.type2lbl = {r: c for c, r in enumerate(self.dataset)}
        self.lbl2type = {c: r for c, r in enumerate(self.dataset)}
        self.triple_text = ' [IISS RREELLAATTEEDD TTOO] '
        self.negative_label = 'τRRAANNDDOOMMτ'

    # this function removes extra information exist in SVOs such as ... via, ... using, etc.
    def sanitize_dataset(self, dataset, stopwords=[]):
        if not stopwords:
            stopwords = ['using','via','that','such as', ', such as','where']
        for typ in dataset:
            for cid in dataset[typ]:
                for idx, rec in enumerate(dataset[typ][cid]):
                    for sw in stopwords:
                        temp_rec = rec.split(" ")
                        rec = rec.split(" "+sw)[0]

                    dataset[typ][cid][idx] = rec
        return dataset

    def return_rules(self):
        rules = {c: [] for c in self.dataset}
        print(">>\tNo rules have been imposed in creating dataset.")
        # rules['Read Files'].extend(
        #     ['Read From Memory', 'Memory Read (Memory Buffer Errors,Pointer Issues,Type Errors,etc.)'])
        # rules['Read From Memory'].extend(
        #     ['Read Files', 'Memory Read (Memory Buffer Errors,Pointer Issues,Type Errors,etc.)'])
        # rules['Memory Read (Memory Buffer Errors,Pointer Issues,Type Errors,etc.)'].extend(
        #     ['Read Files', 'Read From Memory'])

        # rules['Obtain Sensitive Information: Other Data'].extend(['Obtain Sensitive Information: Credentials'])
        # rules['Obtain Sensitive Information: Credentials'].extend(['Obtain Sensitive Information: Other Data'])
        return rules

    def get_negative_pool(self, sample_pool, negative_ids, max_samples=0):
        import random
        neg_pool = []
        if max_samples > 0:
            min_size = max_samples
        else:
            min_size = min([len(sample_pool[id_]) for id_ in negative_ids])
        for id_ in negative_ids:
            neg_pool.extend(random.sample(sample_pool[id_], min(min_size, len(sample_pool[id_]))))
        return neg_pool

    def _add_negative(self, text, sample_pool, negative_ids, max_samples=0, sampling_rate=10):
        lst = []
        import random
        if type(negative_ids) == set or type(negative_ids) == list:
            neg_pool = self.get_negative_pool(sample_pool, negative_ids, max_samples=max_samples)
        else:
            neg_pool = func_merge_listOflist(list(negative_ids.values()))
        neg_samples = random.sample(neg_pool, min(len(neg_pool), sampling_rate))
        for r in neg_samples:
            lst.append([text, r])
        return lst

    def create_dataset(self, CVE_descs, neg_rate=15, max_positive=300, continous_class=True, include_cve_desc=True,
                       shuffle=True,
                       cve_sampling_size=30):
        from sklearn.model_selection import train_test_split
        rules = self.return_rules()

        all_types = list(self.dataset.keys())
        if self.Config.multilabel:
            num_classes = len(all_types) + 1
            print("Number of Classes (Multilabel):", num_classes)
        else:
            num_classes = len(all_types)
            print("Number of Classes (Multiclass):", num_classes)

        sample_pool = {id_: list(set(func_merge_listOflist(self.dataset[id_].values()))) for id_ in self.dataset}
        train_dataset, test_dataset = [[], []]
        pos_count, neg_count = [0, 0]
        for func_type in sample_pool:
            pos_pool = sample_pool[func_type]
            negative_typeIDs = set(all_types).difference(set([func_type] + rules.get(func_type)))

            train_pool, test_pool = train_test_split(pos_pool, test_size=0.2, random_state=13)
            if max_positive > 0:
                train_pool = random.sample(train_pool, min(len(train_pool), max_positive))
            # - Positive sampling
            train_pos = [[r1, r2] for i, r1 in enumerate(train_pool) for r2 in train_pool[i + 1:]]
            pos_count += len(train_pos)
            test_pos = [[r1, r2] for i, r1 in enumerate(test_pool) for r2 in test_pool[i + 1:]]

            # - Negative sampling
            train_neg = list()
            test_neg = list()
            for train_txt in train_pool:
                train_neg.extend(self._add_negative(train_txt, sample_pool, negative_typeIDs, sampling_rate=neg_rate))
            neg_count += len(train_neg)
            for test_txt in train_pool:
                test_neg.extend(self._add_negative(test_txt, sample_pool, negative_typeIDs, sampling_rate=neg_rate))

            # - Create Input Examples (Positive)
            for r_pos in train_pos:
                if continous_class:
                    lbl = 1.0
                else:
                    if self.Config.multilabel:
                        lbl = make_multihot(self.type2lbl[func_type], num_classes)
                        # r_pos = r_pos.append(func_type)
                    else:
                        lbl = self.type2lbl[func_type]
                train_dataset.append(InputExample(texts=r_pos, label=lbl))

            for rt_pos in test_pos:
                if continous_class:
                    lbl = 1.0
                else:
                    if self.Config.multilabel:
                        lbl = make_multihot(self.type2lbl[func_type], num_classes)
                        # rt_pos = rt_pos.append(func_type)
                    else:
                        lbl = self.type2lbl[func_type]
                test_dataset.append(InputExample(texts=rt_pos, label=lbl))

            if include_cve_desc:
                cve_pool = random.sample(list(self.dataset[func_type].keys()), min(cve_sampling_size,
                                                                                   len(list(self.dataset[
                                                                                                func_type].keys()))))
                if continous_class:
                    lbl = 1.0
                else:
                    if self.Config.multilabel:
                        lbl = make_multihot(self.type2lbl[func_type], num_classes)
                    else:
                        lbl = self.type2lbl[func_type]

                for cveid in cve_pool:
                    try:

                        train_dataset.append(
                            InputExample(texts=[self.dataset[func_type][cveid][0], CVE_descs[cveid]], label=lbl))
                    except:
                        print("CVE ID not in the dataset: Ignored!")
                        continue
            # - Create Input Examples (Negative)
            if continous_class:
                for r_neg in train_neg:
                    train_dataset.append(InputExample(texts=r_neg, label=0.0))

                for rt_neg in test_neg:
                    test_dataset.append(InputExample(texts=rt_neg, label=0.0))
            else:
                print(
                    "****************************\nWARNING: With the current settings, the problem must be IR.\nThe negative samples have not been included.")

        if shuffle:
            random.shuffle(train_dataset)
        # logging.warning("Train Positive Samples:\t", pos_count)
        # logging.warning("Train Negative Samples:\t", neg_count)
        print("Train Positive Samples:\t", pos_count)
        print("Train Negative Samples:\t", neg_count)
        return train_dataset, test_dataset

    @staticmethod
    def _concat_triple(lst, to_add, connect_text=' [IS RELATED TO] ', index=-1):
        lst[index] = lst[index] + connect_text + " " + to_add
        return lst

    def nltk_tag_to_wordnet_tag(self, nltk_tag):
        import nltk
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatize_sentence(self, sentence):
        import nltk
        lemmatizer = nltk.WordNetLemmatizer()
        # tokenize the sentence and find the POS tag for each token
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
        # tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], self.nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:
                # else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)

    def remove_entities(self, sent, with_spacy=False):
        if with_spacy:
            import spacy
            print(">> remove entities")
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(sent)
            ents = doc.ents
            for ent in ents:
                sent = sent.replace(str(ent), "#")
        else:
            import re
            r = re.compile(r'[\d.]+\d')
            sent = r.sub("#", sent)
        return sent

    def sent_clean(self, sent, lemma=True, remove_ents=True):
        sent = sent.replace("  ", " ")
        sent = sent.replace(" - ", "-")
        sent = sent.replace(" , ", ", ")
        sent = sent.replace(" ( ", " (")
        sent = sent.replace(" ) ", ") ")
        sent = sent.replace(" / ", "/")
        sent = sent.replace(" i d ", " id ")
        sent = sent.replace(" 's ", "'s ")
        sent = sent.replace("** DISPUTED **  ", "")
        sent = sent.replace("** REVOKED **  ", "")
        sent = sent.lower()
        if lemma:
            sent = self.lemmatize_sentence(sent)
        if remove_ents:
            sent = self.remove_entities(sent)
        return sent

    def longer_first(self, lst):
        if len(lst[0].split(" "))> len(lst[1].split(" ")):
            lst = [lst[1], lst[0]]
        return lst

    # this combines the org dataset with manual dataset in which it randomly pairs the positive samples
    # and pairs the org positives with manual negatives
    def create_dataset_ContextBase(self, CVE_descs, dataset_=None, manual_dataset=False, neg_rate=15, test_size=0.2,
                                   negative_sampling=True, max_positive=300, continuous_class=None, multilabel=None,
                                   include_cve_desc=True, shuffle=True, cve_sampling_size=30, triple_input=True,
                                   add_random_class=False, global_neg_sampling=True):

        from sklearn.model_selection import train_test_split
        rules = self.return_rules()
        # -
        if dataset_ == None:
            dataset = self.dataset
        else:
            dataset = dataset_
        all_types = list(dataset.keys())
        if continuous_class == None:
            continuous_class = self.Config.relevance

        if manual_dataset == None:  # default dataset has no internal negative samples
            global_neg_sampling = True

        if continuous_class:
            num_classes = 1
            multilabel = False
            print("Number of Classes (IR):", num_classes)


        else:
            if add_random_class:
                num_classes = len(all_types) + 1
                print("Number of Classes (Classification) including random class:", num_classes)
            else:
                num_classes = len(all_types)
                print("Number of Classes (Classification) with no random class:", num_classes)

        if manual_dataset:
            print(">>\t'Manual Dataset' is being processed")
            dataset = {f: {cl: {c: [self.sent_clean(s, lemma=True, remove_ents=True) for s in dataset[f][cl][c]]
                                for c in dataset[f][cl]
                                } for cl in dataset[f]
                           } for f in dataset
                       }
            sample_pool = {id_: list(set(func_merge_listOflist(dataset[id_]['pos'].values()))) for id_ in dataset}


        else:
            print(">>\t'Automatic Dataset' is being processed")
            dataset = {f: {c: [
                self.sent_clean(s, lemma=True, remove_ents=True) for s in dataset[f][c]
            ] for c in dataset[f]} for f in dataset}
            sample_pool = {id_: list(set(func_merge_listOflist(dataset[id_].values()))) for id_ in dataset}

        train_dataset, test_dataset = [[], []]
        pos_count, neg_count = [0, 0]

        f_ = True
        for func_type in sample_pool:
            pos_pool = sample_pool[func_type]
            if global_neg_sampling:
                if f_:
                    print(">> Global Negative Sampling...")
                    f_ = False
                negative_typeIDs = set(all_types).difference(
                    set([func_type] + rules.get(func_type)))  # --> It is Type IDs

            else:
                if f_:
                    print(">> Pre-defined Negative Sampling...")
                    f_ = False
                negative_typeIDs = {i: v for i, v in enumerate(
                    list(set(func_merge_listOflist(dataset[func_type]['neg'].values()))))}  # --> It is a set of samples

            train_pool, test_pool = train_test_split(pos_pool, test_size=test_size, random_state=13)
            # print("Test train split done")

            if not max_positive or max_positive < 1:
                max_positive = len(train_pool)-1
                # print("max_positive")
            # - Positive sampling
            train_pos = [self.longer_first([r1, r2]) for i, r1 in enumerate(train_pool) for r2 in random.sample(train_pool, min(len(train_pool), max_positive))]
            pos_count += len(train_pos)
            test_pos = [self.longer_first([r1, r2]) for i, r1 in enumerate(test_pool) for r2 in random.sample(test_pool, min(len(test_pool), max_positive))]
            # print("Positive sampling")
            # - Negative sampling
            train_neg = list()
            test_neg = list()
            for train_txt in train_pool:
                train_neg.extend(self._add_negative(train_txt, sample_pool, negative_typeIDs, sampling_rate=min(len(train_txt), neg_rate)))
            neg_count += len(train_neg)
            for test_txt in train_pool:
                test_neg.extend(self._add_negative(test_txt, sample_pool, negative_typeIDs, sampling_rate=min(len(test_txt), neg_rate)))
            # print("Negative sampling")
            # - Create Input Examples (Positive)
            for r_pos in train_pos:
                # r_pos = self.longer_first(r_pos)
                if continuous_class:
                    lbl = 1.0
                else:
                    if multilabel:
                        lbl = make_multihot(self.type2lbl[func_type], num_classes)
                    else:
                        lbl = self.type2lbl[func_type]

                if triple_input:
                    # r_pos.append(func_type)
                    r_pos = self._concat_triple(r_pos, func_type, connect_text=self.triple_text, index=-1)
                train_dataset.append(InputExample(texts=r_pos, label=lbl, guid="positive"))

            for rt_pos in test_pos:
                # rt_pos = self.longer_first(rt_pos)
                if continuous_class:
                    lbl = 1.0
                else:
                    if multilabel:
                        lbl = make_multihot(self.type2lbl[func_type], num_classes)
                    else:
                        lbl = self.type2lbl[func_type]
                if triple_input:
                    # rt_pos.append(func_type)
                    rt_pos = self._concat_triple(rt_pos, func_type, connect_text=self.triple_text,
                                                 index=-1)
                test_dataset.append(InputExample(texts=rt_pos, label=lbl, guid=func_type))

            if include_cve_desc:
                if manual_dataset:
                    cve_pool = random.sample(list(dataset[func_type]['pos'].keys()),
                                             min(cve_sampling_size, len(list(dataset[func_type]['pos'].keys()))))
                else:
                    cve_pool = random.sample(list(dataset[func_type].keys()),
                                             min(cve_sampling_size, len(list(dataset[func_type].keys()))))
                if continuous_class:
                    lbl = 1.0
                else:
                    if multilabel:
                        lbl = make_multihot(self.type2lbl[func_type], num_classes)
                    else:
                        lbl = self.type2lbl[func_type]

                for cveid in cve_pool:
                    try:
                        text = [dataset[func_type][cveid][0], self.sent_clean(CVE_descs[cveid])]
                        if triple_input:
                            # text.append(func_type)
                            text = self._concat_triple(text, func_type, connect_text=self.triple_text,
                                                       index=-1)
                        train_dataset.append(InputExample(texts=text, label=lbl, guid=func_type))
                    except:
                        print("CVE ID not in the dataset: Ignored!")
                        continue

            # - Create Input Examples (Negative)
            if negative_sampling:
                # - Label Generation (Train)
                if continuous_class:
                    lbl = 0.0
                    negTopic_train = self.negative_label
                elif add_random_class:
                    lbl = num_classes-1
                    negTopic_train = self.negative_label
                else:
                    lbl = random.choice(list(set(range(num_classes)).difference({self.type2lbl[func_type]})))
                    negTopic_train = self.lbl2type[lbl]
                if multilabel:
                    lbl = make_multihot(lbl, num_classes)

                # - InputExample Generation (Train)
                for r_neg in train_neg:
                    if triple_input:
                        # r_neg.append(negTopic_train)
                        r_neg = self._concat_triple(r_neg, negTopic_train, connect_text=self.triple_text,
                                                    index=-1)
                    train_dataset.append(InputExample(texts=r_neg, label=lbl, guid="negative"))

                # - Label Generation (Test)
                if continuous_class:
                    lbl = 0.0
                    negTopic_test = self.negative_label
                elif add_random_class:
                    lbl = num_classes-1
                    negTopic_test = self.negative_label
                else:
                    lbl = random.choice(list(set(range(num_classes)).difference({self.type2lbl[func_type]})))
                    negTopic_test = self.lbl2type[lbl]
                if multilabel:
                    lbl = make_multihot(lbl, num_classes)

                for rt_neg in test_neg:
                    if triple_input:
                        rt_neg.append(negTopic_test)
                        rt_neg = self._concat_triple(rt_neg, negTopic_test, connect_text=self.triple_text,
                                                     index=-1)
                    test_dataset.append(InputExample(texts=rt_neg, label=lbl, guid="negative"))
            else:
                print(
                    "****************************\nWARNING: With the current settings, the problem must be IR.\nThe negative samples have not been included.")

        if shuffle:
            random.shuffle(train_dataset)
        # logging.warning("Train Positive Samples:\t", pos_count)
        # logging.warning("Train Negative Samples:\t", neg_count)
        print("Train Positive Samples:\t", pos_count)
        print("Train Negative Samples:\t", neg_count)
        return train_dataset, test_dataset, num_classes


    def CombineDataset(self, dataset_manual, neg_rate=15, test_size=0.2, negative_sampling=True, max_positive=300,
                       continuous_class=None, multilabel=None, shuffle=True, triple_input=True, add_random_class=False):
        # combines two given dataset, pos of org with manual plus pos of or with neg of manual
        from sklearn.model_selection import train_test_split
        rules = self.return_rules()
        # -

        dataset1 = self.dataset
        all_types = list(dataset1.keys())

        if continuous_class == None:
            continuous_class = self.Config.relevance

        if continuous_class:
            num_classes = 1
            multilabel = False
            print("Number of Classes (IR):", num_classes)

        else:
            if add_random_class:
                num_classes = len(all_types) + 1
                print("Number of Classes (Classification) including random class:", num_classes)
            else:
                num_classes = len(all_types)
                print("Number of Classes (Classification) with no random class:", num_classes)


        print(">>\t'Automatic Dataset' is being processed")
        dataset1 = {f: {c: [
            self.sent_clean(s, lemma=True, remove_ents=True) for s in dataset1[f][c]
        ] for c in dataset1[f]} for f in dataset1}
        sample_pool1 = {id_: list(set(func_merge_listOflist(dataset1[id_].values()))) for id_ in dataset1}

        print(">>\t'Manual Dataset' is being processed")
        dataset2 = {f: {cl: {c: [self.sent_clean(s, lemma=True, remove_ents=True) for s in dataset_manual[f][cl][c]]
                            for c in dataset_manual[f][cl]
                            } for cl in dataset_manual[f]
                       } for f in dataset_manual
                   }
        sample_pool2 = {id_: list(set(func_merge_listOflist(dataset_manual[id_]['pos'].values()))) for id_ in dataset2}
        negative_pool = {id_: list(set(func_merge_listOflist(dataset_manual[id_]['neg'].values()))) for id_ in dataset2}



        train_dataset, test_dataset = [[], []]
        pos_count, neg_count = [0, 0]

        for func_type in sample_pool1:
            pos_pool = sample_pool1[func_type]
            pos_pool2 = sample_pool2[func_type]

            negative_typeIDs = set(all_types).difference(
                set([func_type] + rules.get(func_type)))  # --> It is Type IDs

            train_pool, test_pool = train_test_split(pos_pool, test_size=test_size, random_state=13)
            train_pool2, _ = train_test_split(pos_pool2, test_size=0.1, random_state=13)

            if not max_positive or max_positive < 1:
                max_positive = len(train_pool)-1

            # - Positive sampling
            train_pos = [self.longer_first([r1, r2]) for i, r1 in enumerate(train_pool) for r2 in random.sample(train_pool2, min(len(train_pool2), max_positive))]
            pos_count += len(train_pos)
            test_pos = [self.longer_first([r1, r2]) for i, r1 in enumerate(test_pool) for r2 in random.sample(train_pool2, min(len(train_pool2), max_positive))]
            # print("Positive sampling")

            # - Negative sampling
            train_neg = list()
            test_neg = list()
            for train_txt in train_pool:
                train_neg.extend(self._add_negative(train_txt, negative_pool, [func_type], sampling_rate=neg_rate))
            neg_count += len(train_neg)
            for test_txt in train_pool:
                test_neg.extend(self._add_negative(test_txt, negative_pool, [func_type], sampling_rate=neg_rate))
            # print("Negative sampling")
            # - Create Input Examples (Positive)
            for r_pos in train_pos:
                # r_pos = self.longer_first(r_pos)
                if continuous_class:
                    lbl = 1.0
                else:
                    if multilabel:
                        lbl = make_multihot(self.type2lbl[func_type], num_classes)
                    else:
                        lbl = self.type2lbl[func_type]

                if triple_input:
                    # r_pos.append(func_type)
                    r_pos = self._concat_triple(r_pos, func_type, connect_text=self.triple_text, index=-1)
                train_dataset.append(InputExample(texts=r_pos, label=lbl, guid=func_type))

            for rt_pos in test_pos:
                # rt_pos = self.longer_first(rt_pos)
                if continuous_class:
                    lbl = 1.0
                else:
                    if multilabel:
                        lbl = make_multihot(self.type2lbl[func_type], num_classes)
                    else:
                        lbl = self.type2lbl[func_type]
                if triple_input:
                    # rt_pos.append(func_type)
                    rt_pos = self._concat_triple(rt_pos, func_type, connect_text=self.triple_text,
                                                 index=-1)
                test_dataset.append(InputExample(texts=rt_pos, label=lbl, guid=func_type))

            # - Create Input Examples (Negative)
            if negative_sampling:
                # - Label Generation (Train)
                if continuous_class:
                    lbl = 0.0
                    negTopic_train = self.negative_label
                elif add_random_class:
                    lbl = num_classes-1
                    negTopic_train = self.negative_label
                else:
                    lbl = random.choice(list(set(range(num_classes)).difference({self.type2lbl[func_type]})))
                    negTopic_train = self.lbl2type[lbl]
                if multilabel:
                    lbl = make_multihot(lbl, num_classes)

                # - InputExample Generation (Train)
                for r_neg in train_neg:
                    if triple_input:
                        # r_neg.append(negTopic_train)
                        r_neg = self._concat_triple(r_neg, negTopic_train, connect_text=self.triple_text,
                                                    index=-1)
                    train_dataset.append(InputExample(texts=r_neg, label=lbl, guid="negative"))

                # - Label Generation (Test)
                if continuous_class:
                    lbl = 0.0
                    negTopic_test = self.negative_label
                elif add_random_class:
                    lbl = num_classes-1
                    negTopic_test = self.negative_label
                else:
                    lbl = random.choice(list(set(range(num_classes)).difference({self.type2lbl[func_type]})))
                    negTopic_test = self.lbl2type[lbl]
                if multilabel:
                    lbl = make_multihot(lbl, num_classes)

                for rt_neg in test_neg:
                    if triple_input:
                        rt_neg.append(negTopic_test)
                        rt_neg = self._concat_triple(rt_neg, negTopic_test, connect_text=self.triple_text,
                                                     index=-1)
                    test_dataset.append(InputExample(texts=rt_neg, label=lbl, guid="negative"))
            else:
                print(
                    "****************************\nWARNING: With the current settings, the problem must be IR.\nThe negative samples have not been included.")

        if shuffle:
            random.shuffle(train_dataset)
        # logging.warning("Train Positive Samples:\t", pos_count)
        # logging.warning("Train Negative Samples:\t", neg_count)
        print("Train Positive Samples:\t", pos_count)
        print("Train Negative Samples:\t", neg_count)
        return train_dataset, test_dataset, num_classes

    def create_SOV2CVE_dataset(self,CVE_descs, dset_list, max_positive=300, continuous_class=None, multilabel=None, shuffle=True,
                               add_random_class=False, test_size=0.1):
        from sklearn.model_selection import train_test_split
        import random
        all_types = list(dset_list[0].keys())
        if continuous_class:
            num_classes = 1
            multilabel = False
            print("Number of Classes (IR):", num_classes)

        else:
            if add_random_class:
                num_classes = len(all_types) + 1
                print("Number of Classes (Classification) including random class:", num_classes)
            else:
                num_classes = len(all_types)
                print("Number of Classes (Classification) with no random class:", num_classes)

        train_dataset, test_dataset = [[], []]
        for typ in all_types:
            if continuous_class:
                lbl = 1.0
            else:
                if multilabel:
                    lbl = make_multihot(self.type2lbl[typ], num_classes)
                else:
                    lbl = self.type2lbl[typ]

            for dset in dset_list:
                cveid_list = list(dset[typ].keys())
                cveids_train, cveids_test = train_test_split(cveid_list, test_size=test_size, random_state=13)
                cveids_train = random.sample(cveids_train, min(len(cveids_train), max_positive))

                #------- Training set
                for cveid in cveids_train:
                    content_list = dset[typ][cveid]
                    context = CVE_descs.get(cveid)
                    if context:
                        context = self.sent_clean(context)
                        for content in content_list:
                            train_dataset.append(InputExample(texts=[content, context], label=lbl, guid=typ))

                #------- Testing set
                for cveid in cveids_test:
                    content_list = dset[typ][cveid]
                    context = CVE_descs.get(cveid)
                    if context:
                        context = self.sent_clean(context)
                        for content in content_list:
                            test_dataset.append(InputExample(texts=[content, context], label=lbl, guid=typ))
        if shuffle:
            random.shuffle(train_dataset)
        return train_dataset, test_dataset



    # def create_multilable(self):

def log_specifications(arg, dir=os.path.join(os.getcwd(), "Models"), fname=None, save_log=True):
    import random
    try:
        from datetime import datetime
        dt = datetime.now().strftime("%d-%m-%Y %H-%M")
    except:
        dt = 'NA_' + str(random.choice(list(range(1, 5000)))) + "_" + str(random.choice(list(range(1,50))))

    d = {'dataset_': arg.dataset_,
         'manual_dataset': arg.manual_dataset,
         'neg_rate': arg.neg_rate,
         'test_size': arg.test_size,
         'negative_sampling': arg.negative_sampling,
         'max_positive': arg.max_positive,
         'continuous_class': arg.continuous_class,
         'multilabel': arg.multilabel,
         'include_cve_desc': arg.include_cve_desc,
         'shuffle': arg.shuffle,
         'cve_sampling_size': arg.cve_sampling_size,
         'triple_input': arg.triple_input,
         'add_random_class': arg.add_random_class,
         'global_neg_sampling': arg.global_neg_sampling}
    print(d)
    if fname == None:
        fname = "_".join([dt,"SVO2Func" 
                        "negr"+str(d['neg_rate']),
                        "ngsmpl"+str(d['negative_sampling']),
                        "cntcls"+str(d['continuous_class']),
                        "mltlbl"+str(d['multilabel']),
                        "trplipt"+str(d['triple_input']),
                        "rndsmpl"+str(d['add_random_class'])])
    else:
        fname = "_".join([dt, "SVE2Func"])

    foname = os.path.join(dir, fname)
    if not os.path.isfile(foname):
        os.mkdir(foname)
    file = os.path.join(foname, "specifications.txt")

    with open(file, "w", encoding='utf-8') as f:
        for ar in d:
            f.write(ar + ": " + str(d[ar]))
            f.write("\n")
    return foname


from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from sentence_transformers import util,SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator

logger = logging.getLogger(__name__)


class CrossEncoder():
    def __init__(self, model_name: str, num_labels: int = None, max_length: int = None, device: str = None,
                 tokenizer_args: Dict = {},
                 default_activation_function=None, multilabel=False):
        """
        A CrossEncoder takes exactly two sentences / texts as input and either predicts
        a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
        on a scale of 0 ... 1.

        It does not yield a sentence embedding and does not work for individually sentences.

        :param model_name: Any model name from Huggingface Models Repository that can be loaded with AutoModel. We provide several pre-trained CrossEncoder models that can be used for common tasks
        :param num_labels: Number of labels of the classifier. If 1, the CrossEncoder is a regression model that outputs a continous score 0...1. If > 1, it output several scores that can be soft-maxed to get probability scores for the different classes.
        :param max_length: Max length for input sequences. Longer sequences will be truncated. If None, max length of the model will be used
        :param device: Device that should be used for the model. If None, it will use CUDA if available.
        :param tokenizer_args: Arguments passed to AutoTokenizer
        :param default_activation_function: Callable (like nn.Sigmoid) about the default activation function that should be used on-top of model.predict(). If None. nn.Sigmoid() will be used if num_labels=1, else nn.Identity()
        """
        self.multilabel = multilabel
        self.config = AutoConfig.from_pretrained(model_name)
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
            except Exception as e:
                logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)))
        elif hasattr(self.config,
                     'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
            self.default_activation_function = util.import_from_string(
                self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt",
                                   max_length=self.max_length)

        labels = torch.tensor(labels,
                              dtype=torch.float if self.config.num_labels == 1 or self.multilabel else torch.long).to(
            self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt",
                                   max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized

    def fit(self,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct=None,
            activation_fct=nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                                           t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05,
                                         disable=not show_progress_bar):
                if use_amp:
                    with autocast():
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

    def predict(self, sentences: List[List[str]],
                batch_size: int = 32,
                show_progress_bar: bool = None,
                num_workers: int = 0,
                activation_fct=None,
                apply_softmax=False,
                convert_to_numpy: bool = True,
                convert_to_tensor: bool = False
                ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.

        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only,
                                    num_workers=num_workers, shuffle=False)

        if show_progress_bar is None:
            show_progress_bar = (
                        logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path):
        """
        Same function as save
        """
        return self.save(path)


def compute_metrics(labels_, preds):
    # labels = pred.label_ids
    preds = preds.argmax(-1)
    # preds = [p.argmax(-1) for p in preds]

    precision_mi, recall_mi, f1_mi, _ = precision_recall_fscore_support(labels_, preds, average='micro')
    precision_ma, recall_ma, f1_ma, _ = precision_recall_fscore_support(labels_, preds, average='macro')
    acc = accuracy_score(labels_, preds)

    cm = confusion_matrix(labels_, preds)
    # plt.show()
    res = {
        'accuracy': acc,
        'f1 (micro)': f1_mi,
        'precision (micro)': precision_mi,
        'recall (micro)': recall_mi,
        'f1 (macro)': f1_ma,
        'precision (macro)': precision_ma,
        'recall (macro)': recall_ma,
        'confusion matrix': cm,
    }

    print(str(res))
    return res
