"""Platform-compatible distilled foundation fills plus compact energy blend.

Research fit requires TabICL; deployed artifact contains only plain arrays,
metadata and LightGBM booster text. Fixed half blends independently evaluated
in five-fold feature holdout and three repeated complete-energy holdouts.
"""
import hashlib
import joblib
import numpy as np
import pandas as pd
import torch
from common import ROOT,test_mask_patterns
from candidates.c18_private_safe import PREDICT_CODE
from exp53_distill_tabicl import fit_student


def fit_artifact(train):
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.72)
    source=ROOT/"package/c21_compact_energy_blend/model.pkl";art=dict(joblib.load(source))
    art["imputers"]={c:list(models) for c,models in art["imputers"].items()}
    test=pd.read_csv(ROOT/"test.csv");X=train.drop(columns="energy_usage")
    rows=pd.concat([X,test[X.columns]],ignore_index=True);dom=np.r_[np.zeros(len(X)),np.ones(len(test))]
    students,tables,debug=fit_student(rows,dom,test_mask_patterns(test),copies=5,trio=True,synthetic=True)
    assert tables==art["dev_tables"]
    for col,models in students.items():
        original_count=len(art["imputers"][col]);assert original_count%len(models)==0
        # Equal total members on each side preserve the validated 50/50 blend.
        art["imputers"][col].extend(models*(original_count//len(models)))
    art["source_candidate"]="c21_compact_energy_blend"
    art["source_artifact_sha256"]=hashlib.sha256(source.read_bytes()).hexdigest()
    art["experimental_change"]=dict(features=list(students),student_blend=.5,energy_compact_blend=.5,
                                     teacher="TabICL v2, 8 estimators",synthetic_query_seed=713,
                                     mask_copies=5,student_training_debug=debug,
                                     energy_labels_used_by_teacher=False)
    return art
