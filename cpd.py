import persim
from gtda.time_series import TakensEmbedding

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm
import ruptures as rpt
from scipy import interpolate
from collections import Counter
from roerich import change_point
from scipy.stats import bootstrap
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.model_selection import train_test_split, KFold#Stratified
from torch.distributions import MultivariateNormal as MNormal
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def autoregression_matrix(X, periods=1, fill_value=0):
    shifted_x = [pd.DataFrame(X).shift(periods=i, fill_value=fill_value).values for i in range(periods)]
    X_auto = np.hstack(tuple(shifted_x))
    return X_auto


def ref_test_windows(X, window_size=10, step=1):
    T = []
    ref = []
    test = []
    for i in range(2 * window_size - 1, len(X), step):
        T.append(i)
        ref.append(X[i - 2 * window_size + 1 : i - window_size + 1])
        test.append(X[i - window_size + 1 : i + 1])
    return np.array(T), np.array(ref), np.array(test)


def KL_sym(ref_preds, test_preds):
    # test_preds[test_preds<0], ref_preds[ref_preds<0] = 0, 0
    return np.mean(np.log(test_preds + 1e-5))     - np.mean(np.log(1. - test_preds + 1e-5)) + \
           np.mean(np.log(1. - ref_preds + 1e-5)) - np.mean(np.log(ref_preds + 1e-5))


def unified_score(T, T_score, score):
    uni_score = np.zeros(len(T))
    interp = interpolate.interp1d(T_score, score, kind='previous', fill_value=(0, 0), bounds_error=False)
    uni_score = interp(T)
    return uni_score


class GenModCPD(object):
    def __init__(self, GenMod, window_size=50, step=1, periods=1, clf='QDA', scale=True):
        
        self.GenMod      = GenMod
        self.window_size = window_size
        self.step        = step
        self.periods     = periods
        self.clf         = clf
        self.scale       = scale
        
        
        
    def _scale(self, X_ref, X_test):
        
        ss = StandardScaler()
        ss.fit(np.concatenate((X_ref, X_test), axis=0))
        X_ref_ss = ss.transform(X_ref)
        X_test_ss = ss.transform(X_test)
        
        return X_ref_ss, X_test_ss
    
    
    def _default_exam(self, X_gen, X):

        XX = np.concatenate((X_gen, X), axis=0)
        yy = np.array([0] * len(X_gen) + [1] * len(X))
        
        if self.clf == 'QDA':
            self.classifier = QuadraticDiscriminantAnalysis()
        elif self.clf == 'DT':
            self.classifier = DecisionTreeClassifier()
        elif self.clf == 'MLP':
            self.classifier = MLP(activation='tanh', hidden_layer_sizes = (16,),
                             learning_rate_init=1e-4, max_iter=100, batch_size=self.window_size, random_state=73)
        
        pred_top_gen, pred_top = [], []
        if len(XX) < 22:
            n_splits = 2
        else:
            n_splits = 10
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=73)
        
        for train_index, test_index in cv.split(XX, yy):
            XX_train, XX_test = XX[train_index], XX[test_index]
            yy_train, yy_test = yy[train_index], yy[test_index]
            
            self.classifier.fit(XX_train, yy_train)
            yy_pred = self.classifier.predict_proba(XX_test)[:, 1]
                
            pred_top_gen.append(yy_pred[yy_test==0])
            pred_top.append(yy_pred[yy_test==1])
        
        return np.concatenate(pred_top_gen), np.concatenate(pred_top)
    
    
    def one_step_predict(self, X_ref, X_test):
        
        if self.scale:
            X_ref, X_test = self._scale(X_ref, X_test)
        
        self.GenMod.fit(X_ref)
        X_gen_ref, y = self.GenMod.sample(self.window_size)
        
        pred_gen_ref, pred_test = self._default_exam(X_gen_ref, X_test)
        score = KL_sym(pred_gen_ref, pred_test)
        
        return score
    
    
    def predict(self, X):
        X_auto = autoregression_matrix(X, periods=self.periods, fill_value=0)
        
        T, ref, test = ref_test_windows(X_auto, window_size=self.window_size, step=self.step)
        
        scores = []
        T_scores = []
        iters = range(0, len(ref))
        for i in tqdm(iters):
            
            ascore = self.one_step_predict(ref[i], test[i])
            scores.append(ascore)
        
        T_scores = np.array([T[i] for i in iters])
        
        T = np.arange(len(X))
        
#         print(T_scores.shape, np.array(scores).shape)
        scores = np.array(scores).squeeze()
        
        scores = unified_score(T, T_scores, scores)
        scores = unified_score(T, T - self.window_size, scores)
        
        return np.array(scores)

    
##################################################
###                                            ###
###                 CPD TDA                    ###
###                                            ###
##################################################

        
class TDACPD(GenModCPD):
    
    def __init__(self, persistence, window_size=50, step=1, periods=1, clf='QDA', scale=True, dim=2):
        
        super().__init__(persistence, window_size, step, periods, clf, scale)
        self.embedding = TakensEmbedding(dimension=dim)
        self.dim=dim
        
    
    def one_step_predict(self, X_ref, X_test):
        
        if self.scale:
            X_ref, X_test = self._scale(X_ref, X_test)
            
        X_ref  = np.swapaxes(X_ref, 0, 1)
        X_test = np.swapaxes(X_test, 0, 1)
            
        emb_ref  = self.embedding.fit_transform(X_ref)
        emb_test = self.embedding.fit_transform(X_test)

        dgm_ref  = self.GenMod.fit_transform(emb_ref)
        dgm_test = self.GenMod.fit_transform(emb_test)
        
        dgm_ref = np.delete(dgm_ref[0], -1, 1)
        dgm_test = np.delete(dgm_test[0], -1, 1)
        score = persim.wasserstein(dgm_ref, dgm_test)
        
        return score
    

##################################################
###                                            ###
###                 CPD Multi                  ###
###                                            ###
##################################################

        
class GenModCPDMulti(GenModCPD):
    
    def __init__(self, GenMod, window_size=50, step=1, periods=1, n_exams=1, clf='QDA', scale=True):
        
        super().__init__(GenMod, window_size, step, periods, clf, scale)
        self.n_exams = n_exams
        
    
    def one_step_predict(self, X_ref, X_test):
        
        if self.scale:
            X_ref, X_test = self._scale(X_ref, X_test)
        
        self.GenMod.fit(X_ref)
        
        scores = []
        
        for _ in range(self.n_exams):
            X_gen_ref, y = self.GenMod.sample(self.window_size)

            pred_gen_ref, pred_test = self._default_exam(X_gen_ref, X_test)
            score = KL_sym(pred_gen_ref, pred_test)
            scores.append(score)
        
        score = np.mean(scores)
        
        return score
    

##################################################
###                                            ###
###                 CPD Double                 ###
###                                            ###
##################################################

    
class GenModCPDDouble(GenModCPD):

    def __init__(self, GenMod_ref, GenMod_test, window_size=50, step=1, periods=1, exam_window_size=10000, clf='QDA', scale=True):
        
        super().__init__(GenMod_ref, window_size, step, periods, clf, scale)
        self.GenMod_test = GenMod_test
        self.exam_window_size = exam_window_size
        
        
    def one_step_predict(self, X_ref, X_test):
        
        if self.scale:
            X_ref, X_test = self._scale(X_ref, X_test)
        
        self.GenMod.fit(X_ref)
        self.GenMod_test.fit(X_test)
        
        X_gen_ref, y = self.GenMod.sample(self.exam_window_size)
        X_gen_test, y = self.GenMod_test.sample(self.exam_window_size)

        pred_gen_ref, pred_gen_test = self._default_exam(X_gen_ref, X_gen_test)
        score = KL_sym(pred_gen_ref, pred_gen_test)
        
        return score
    

class AugCPDDouble(GenModCPD):

    def __init__(self, GenMod_ref, GenMod_test, window_size=50, step=1, periods=1, exam_window_size=10000, clf='QDA', scale=True):
        
        super().__init__(GenMod_ref, window_size, step, periods, clf, scale)
        self.GenMod_test = GenMod_test
        self.exam_window_size = exam_window_size
        
        
    def one_step_predict(self, X_ref, X_test):
        
        if self.scale:
            X_ref, X_test = self._scale(X_ref, X_test)
            
        X = np.concatenate((X_test, X_ref))
        y = np.concatenate((np.zeros(self.window_size), np.ones(self.window_size)))
        X_gen_ref, y_gen = self.GenMod.fit_resample(X, y)
        
        X = np.concatenate((X_ref, X_test))
        X_gen_test, y_gen = self.GenMod_test.fit_resample(X, y)
        
        pred_gen_ref, pred_gen_test = self._default_exam(X_gen_ref, X_gen_test)
        score = KL_sym(pred_gen_ref, pred_gen_test)
        
        return score


##################################################
###                                            ###
###           Conditional CPD DGMs             ###
###                                            ###
##################################################

    
class DGMCPD(GenModCPD):

    def __init__(self, GenMod_ref, window_size=50, step=1, periods=1, exam_window_size=10000, clf='QDA', scale=True):
        
        super().__init__(GenMod_ref, window_size, step, periods, clf, scale)
        self.exam_window_size = exam_window_size
        
        
    def one_step_predict(self, X_ref, X_test):
        
        if self.scale:
            X_ref, X_test = self._scale(X_ref, X_test)
            
        cond = np.concatenate((np.zeros((self.window_size, 1)), np.ones((self.window_size, 1))))
        X    = np.concatenate((X_ref, X_test))
        
        self.GenMod.fit(X, cond)
        
        sample_cond = np.concatenate((np.zeros((self.exam_window_size, 1)), np.ones((self.exam_window_size, 1))))
        X_gen = self.GenMod.sample(sample_cond)

        pred_gen_ref, pred_gen_test = self._default_exam(X_gen[:self.exam_window_size], X_gen[self.exam_window_size:])
        score = KL_sym(pred_gen_ref, pred_gen_test)
        
        return score


##################################################
###                                            ###
###                 Exam CPD                   ###
###                                            ###
##################################################


class ExamCPD(GenModCPD):
    
    def __init__(self, window_size=50, step=1, periods=1, clf='QDA', scale=True):
        
        super().__init__(None, window_size, step, periods, clf, scale)
    
    def remove_nans(self, x):
#         x = np.float64(x)
#         nans = np.isnan(x)
#         y = lambda z: z.nonzero()[0]
#         x[nans] = np.interp(y(nans), y(~nans), x[~nans])
        x = pd.Series(x).interpolate().tolist()
        return np.array(x)
    
    def one_step_predict(self, X_ref, X_test):
        
        if self.scale:
            X_ref, X_test = self._scale(X_ref, X_test)
        
        # self.real_ref_windows.append(X_ref)
        # self.real_test_windows.append(X_test)
        
        pred_ref, pred_test = self._default_exam(X_ref, X_test)
        
        pred_ref = self.remove_nans(pred_ref)
        pred_test = self.remove_nans(pred_test)
        
        score = KL_sym(pred_ref, pred_test)
        score = self.remove_nans(score)
        return score


##################################################
###                                            ###
###                 Cost CPD                   ###
###                                            ###
##################################################


class CostDoubleCPD(GenModCPD):
    
    def __init__(self, GenMod_ref, GenMod_test, window_size=50, step=1, periods=1, exam_window_size=1000, scale=True, gamma=1, cost_function='RBF'):
        
        super().__init__(GenMod_ref, window_size, step, periods, scale)
        self.GenMod_test = GenMod_test
        self.exam_window_size = exam_window_size
        self.gamma = gamma
        self.cost_function = cost_function
        
        
    def one_step_predict(self, X_ref, X_test):
        
        # if self.scale:
        #     X_ref, X_test = self._scale(X_ref, X_test)
        
        self.GenMod.fit(X_ref)
        self.GenMod_test.fit(X_test)
        
        X_gen_ref, y = self.GenMod.sample(self.exam_window_size)
        X_gen_test, y = self.GenMod_test.sample(self.exam_window_size)
        
        XX = np.concatenate((X_gen_ref, X_gen_test), axis=0)

        if self.cost_function != 'Energy':
            
            if self.cost_function == 'RBF':
                cost_func = rpt.costs.CostRbf(gamma=self.gamma)
            elif self.cost_function == 'L2':
                cost_func = rpt.costs.CostL2()
            
            cost_func.fit(XX)
            score = cost_func.error(0, len(XX)) - cost_func.error(0, len(X_gen_ref)) - cost_func.error(len(X_gen_ref), len(XX))

        else:

            cost_func = change_point.EnergyDistanceCalculator()
            score = cost_func.reference_test_predict(X_gen_ref, X_gen_test)
        
        return score


class CostExamCPD(GenModCPD):
    
    def __init__(self, window_size=50, step=1, periods=1, scale=True, gamma=1, cost_function='RBF'):
        
        super().__init__(None, window_size, step, periods, scale)
        self.gamma = gamma
        self.cost_function = cost_function
        
        
    def one_step_predict(self, X_ref, X_test):
        
        # if self.scale:
        #     X_ref, X_test = self._scale(X_ref, X_test)
        
        XX = np.concatenate((X_ref, X_test), axis=0)
        
        if self.cost_function != 'Energy':
            
            if self.cost_function == 'RBF':
                cost_func = rpt.costs.CostRbf(gamma=self.gamma)
            elif self.cost_function == 'L2':
                cost_func = rpt.costs.CostL2()
            
            cost_func.fit(XX)
            score = cost_func.error(0, len(XX)) - cost_func.error(0, len(X_ref)) - cost_func.error(len(X_ref), len(XX))

        else:

            cost_func = change_point.EnergyDistanceCalculator()
            score = cost_func.reference_test_predict(X_ref, X_test)
        
        return score
