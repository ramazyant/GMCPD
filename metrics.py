import numpy as np
import pandas as pd
from scipy.signal import argrelmax
from sklearn.metrics import roc_auc_score
from ruptures.metrics import randindex, precision_recall


def find_peaks(score, order):
    cps_pred = argrelmax(score, order=order)[0]
    height_pred = score[cps_pred]
    return cps_pred, height_pred


class OfflineQualityMetrics(object):
    
    def __init__(self, window_size, margin, thresholds):
        self.thresholds = thresholds
        self.window_size = window_size
        self.margin = margin
        
    
    def _metrics(self, cps_true, cps_pred):
        
        n_cr = 0
        n_cp = len(cps_true)
        n_al = len(cps_pred)
        
        if n_al == 0: 
            return 0, 0
        
        if n_cp == 0:
            return 0, 1

        is_used = []
        for atrue in cps_true:
            for apred in cps_pred:
                if (np.abs(apred - atrue) <= self.margin) and (apred not in is_used):
                    n_cr += 1
                    is_used.append(apred)
                    break

        tpr = n_cr / n_cp
        fpr = (n_al - n_cr) / n_al
        
        return tpr, fpr
        
        
    def estimate(self, score, cps_true):
                
        cps_pred, height_pred = find_peaks(score, self.window_size)
        self.cps_pred = cps_pred
        
        data = []
        for thr in self.thresholds:
            
            cps_thr = cps_pred[height_pred > thr]
            tpr, fpr = self._metrics(cps_true, cps_thr)
            recall = tpr
            precision = 1 - fpr
            data.append([thr, tpr, fpr, recall, precision])
            
        data.insert(0, [-999, 1.0, 1.0, 1.0, 0.0])
        data.append([999, 0.0, 0.0, 0.0, 1.0])
        
        curve = pd.DataFrame(columns=['Threshold', 'TruePositiveRate', 'FalsePositiveRate', 'Recall', 'Precision'], 
                             data=data)
        return curve
    
    
    def pr_auc(self, score, cps_true):
        
        curve = self.estimate(score, cps_true)
        
        tpr = curve['TruePositiveRate'].values
        fpr = curve['FalsePositiveRate'].values

        recall = curve['Recall'].values
        precision = curve['Precision'].values
        
        pr_auc = np.abs(np.trapz(precision, recall))
        
        return pr_auc
    
    
    def roc_auc(self, label, score):
        
        T_label = np.arange(len(label))
        T_score = np.arange(len(score))
        
        new_label = np.zeros(len(label))
        T_change = T_label[label == 1]
        for t in T_change:
            cond = (T_label - t < self.window_size) * (T_label - t >= -self.window_size)
            new_label[cond] = 1

        new_label = new_label[T_score]
        roc_auc = roc_auc_score(new_label, score)

        return roc_auc
    
    
    def compute_qm(self, label, score):
        
        cps_true = np.arange(len(label))[label != 0]
        
        # 3. PR AUC
        pr_auc = self.pr_auc(score, cps_true)
        # 4. ROC AUC
#         roc_auc = self.roc_auc(label, score)
        # 1. Ruptures Rand Index
#         cps_true = list(cps_true) + [len(label)]
#         cps_pred = list(self.cps_pred) + [len(label)]
#         ri = randindex(cps_true, cps_pred)
        # 2. Ruptures F1 score
#         precision, recall = precision_recall(cps_true, cps_pred, margin=self.margin)
#         f1 = 2 * precision * recall / (precision + recall + 10**-6)
        
#         return [ri, f1, pr_auc, roc_auc]
        return pr_auc
