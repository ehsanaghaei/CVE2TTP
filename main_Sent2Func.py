import os

import pandas as pd

from Config import Config
from lib.VO2FuncType_functions import CrossEncoder, Create_TrainDataset, func_read_json, merge_dictionaries, CVE_creatdict, CVE_from_NVD
from lib.eval_functions import eval_classifcation

# ============= Initialization
NVD_PATH = "/media/ea/SSD2/DB/Dropbox/ThreatZoom/NVD/"
class FNAMES:
    NVD_PATH = os.path.join(os.getcwd(), 'NVD') + '/'


class arg:
    dataset_ = None
    manual_dataset = False
    neg_rate = 3
    test_size = 0.25
    negative_sampling = True
    max_positive = 80
    continuous_class = False
    multilabel = False
    include_cve_desc = False
    shuffle = True
    cve_sampling_size = 1
    triple_input = False
    add_random_class = True
    global_neg_sampling = True
    if not negative_sampling:
        global_neg_sampling = False
        add_random_class = False


print('\tRunning in local machine')


NVD_list = CVE_from_NVD(NVD_PATH)
# ============== Prepare Dataset
CVE_descs = CVE_creatdict(NVD_list)
dataset = Create_TrainDataset(Config)
pd.DataFrame.from_dict(dataset.lbl2type, 'index').to_csv("vo2func_labels.csv")

dataset_TRs = func_read_json(os.path.join(Config.path_VO_data, 'dataset_manual2.json'))
dataset_TRs = {r: dataset_TRs[r] for r in dataset_TRs if len(dataset_TRs[r]) > 2}
dataset_TRs_paragraphs = func_read_json(os.path.join(Config.path_VO_data, 'dataset_manual2_paragraphs.json'))

dataset.dataset = merge_dictionaries([dataset.dataset, dataset_TRs])

train_dataset, test_dataset, num_classes = dataset.create_dataset_ContextBase(CVE_descs, dataset_=arg.dataset_,
                                                                              manual_dataset=arg.manual_dataset,
                                                                              neg_rate=arg.neg_rate,
                                                                              test_size=arg.test_size,
                                                                              negative_sampling=arg.negative_sampling,
                                                                              max_positive=arg.max_positive,
                                                                              continuous_class=arg.continuous_class,
                                                                              multilabel=arg.multilabel,
                                                                              include_cve_desc=arg.include_cve_desc,
                                                                              shuffle=arg.shuffle,
                                                                              cve_sampling_size=arg.cve_sampling_size,
                                                                              triple_input=arg.triple_input,
                                                                              add_random_class=arg.add_random_class,
                                                                              global_neg_sampling=arg.global_neg_sampling)

# ============== Load Model
model = CrossEncoder(os.path.join("/media/ea/SSD2/DB/Dropbox/BERT_Torch/Models", "13-06-2022 01-43_SVO2Funcnegr3_ngsmplTrue_cntclsFalse_mltlblFalse_trpliptFalse_rndsmplTrue"))
model = CrossEncoder("./models")

eval_classifcation(model, dataset, apply_softmax=False, pairs=False)


