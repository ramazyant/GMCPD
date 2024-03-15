import os
import cpd
import metrics
import numpy as np
from copy import deepcopy
from baseline_methods import *
from scipy.signal import savgol_filter
from imblearn.over_sampling import SMOTE
from gtda.homology import VietorisRipsPersistence
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.spatial.distance import pdist, squareform
from imblearn.over_sampling import RandomOverSampler as ROS
from synth_data_generation import dataset1, dataset2, dataset3

import warnings
warnings.filterwarnings('ignore')


def get_auc(score, label, window_size=50, margin=20, get_curve=False):
    
    cps_true = np.arange(len(label))[label != 0]
    
    qm = metrics.OfflineQualityMetrics(order=window_size, window=margin, thresholds=np.linspace(0, max(score), 1001))
    curve = qm.estimate(score, cps_true)
    auc = qm.auc(score, cps_true)
    
    return auc


class Tester(object):
    def __init__(self, model=None, window_size=50, dataset='Mean', random_seeds=[None]*10, clf='QDA', score_type='KL', exam_window_size=10000, margin=20, scale=True, return_model=False, cond_dgm=False, save_scores=False, model_name=None, random_state=73):
        
        self.model = model
        self.window_size = window_size
        self.dataset = dataset
        self.random_seeds = random_seeds
        self.clf = clf
        self.score_type = score_type
        self.margin = margin
        self.scale = scale
        self.cond_dgm = cond_dgm
        self.save_scores = save_scores
        self.model_name = model_name
        self.random_state=random_state
        
        if score_type == 'RBF': # for computing efficiency
            self.exam_window_size = exam_window_size // 10
        else:
            self.exam_window_size = exam_window_size
        
    
    def load_data(self, file):
    
        data = pd.read_csv(file)
        label = data['Label'].values
        X = data.drop(columns=["Time", "Label"]).values
        X = np.nan_to_num(X)

        return X, label
    
    
    def get_data(self):
        
        XX, labels = [], []
        
        if self.dataset in ['wisdm', 'usc', 'bee', 'hasc']:
        
            dir_path = '../data/' + self.dataset + '/'

            files = []
            for file in sorted(os.listdir(dir_path)):
                if file.endswith('csv'):
                    files.append(dir_path + file)

            for file in files:
                X, label = self.load_data(file)

                XX.append(X)
                labels.append(label)

        else:

            for rs in enumerate(self.random_seeds, start=1):

                # Generate a sample
                if self.dataset == 'Mean':
                    X, label = dataset1(period=100, N_tot=5000, random_seed=rs)
                elif self.dataset == 'Var':
                    X, label = dataset2(period=100, N_tot=5000, random_seed=rs)
                elif self.dataset == 'Cov':
                    X, label = dataset3(period=100, N_tot=5000, random_seed=rs)

                XX.append(X)
                labels.append(label)
        
        return XX, labels
    
    
    def cpd_test(self, XX=None, labels=None):
        
        scores = []
        model_ref, model_test = None, None
        if XX == None:
            XX, labels = self.get_data()
        
        print('Fitting the model...')
        
        for i, X in enumerate(XX, start=1):
        
            print(f'Dataset #{i}')

            if self.model == 'Win':

                score = windows(X, width=self.window_size)
            
            elif self.model == 'KL-CPD':

                score = klcpd(X, window_size=self.window_size)

            elif self.model == 'TIRE':

                score = tire(X, window_size=self.window_size)

            elif self.model == 'RuLSIF':

                score = rulsif(X, window_size=self.window_size)
            
            elif self.model == 'TDA':
            
                persistence = VietorisRipsPersistence(
                    metric="euclidean",
                    homology_dimensions=[0, 1, 2],#[i for i in range(X.shape[1])],#
                    n_jobs=6,
                    collapse_edges=True)
                
                detector = cpd.TDACPD(persistence, window_size=self.window_size, dim=self.exam_window_size)
                score = detector.predict(X)
            
            elif self.model in ['ROS', 'SMOTE']:
                
                if self.model == 'ROS':
                    model_ref  = ROS(sampling_strategy={1: self.exam_window_size}, random_state=self.random_state)
                    model_test = ROS(sampling_strategy={1: self.exam_window_size}, random_state=self.random_state)
                else:
                    model_ref  = SMOTE(sampling_strategy={1: self.exam_window_size}, random_state=self.random_state)
                    model_test = SMOTE(sampling_strategy={1: self.exam_window_size}, random_state=self.random_state)
                
                detector = cpd.AugCPDDouble(model_ref, model_test, window_size=self.window_size, clf=self.clf)
                score = detector.predict(X)
            
            elif self.model == 'Exam':

                if self.score_type == 'KL':
                    detector = cpd.ExamCPD(window_size=self.window_size, clf=self.clf)
                elif self.score_type in ['RBF', 'L2', 'Energy']:
                    gamma = 1./np.median(pdist(X, metric='sqeuclidean'))
                    detector = cpd.CostExamCPD(window_size=self.window_size, gamma=gamma, cost_function=self.score_type)

                score = detector.predict(X)

            else:
                
                model_ref  = deepcopy(self.model)
                model_test = deepcopy(self.model)

                if self.score_type == 'KL':
                    if self.cond_dgm:
                        detector = cpd.DGMCPD(model_ref, window_size=self.window_size, clf=self.clf)
                    else:
                        detector = cpd.GenModCPDDouble(model_ref, model_test, window_size=self.window_size, clf=self.clf)

                elif self.score_type in ['RBF', 'L2', 'Energy']:
                    gamma = 1./np.median(pdist(X, metric='sqeuclidean'))
                    detector = cpd.CostDoubleCPD(model_ref, model_test, window_size=self.window_size, exam_window_size=self.exam_window_size, scale=self.scale, gamma=gamma, cost_function=self.score_type)

                score = detector.predict(X)
            
            score = score + np.random.normal(0, 10**-6, score.shape)
            scores.append(score)

        print('Computing metrics...')

        qms = []

        for s, l in zip(scores, labels):
            qm = metrics.OfflineQualityMetrics(window_size=self.window_size, margin=self.margin, thresholds=np.linspace(0, max(s), 1001))
            qms.append(qm.compute_qm(label=l, score=s))

#         mu = np.mean(qms, axis=0)
#         er = np.std(qms, axis=0) / np.sqrt(len(qms))
#         mu = np.mean
        
        if self.save_scores:
            tbs = np.concatenate((scores, labels), axis=-1)

            if type(self.model) != str:
                self.model = self.model_name

            file_name = f'scores/{self.model}_{self.dataset}_{self.clf}_{self.score_type}.npy'

            np.save(file_name, tbs)
        
#         ri, f1 = "%.3f ± %.3f" % (mu[0], er[0]), "%.3f ± %.3f" % (mu[1], er[1])
#         pr, roc = "%.3f ± %.3f" % (mu[2], er[2]), "%.3f ± %.3f" % (mu[3], er[3])
        pr_auc = "%.3f ± %.3f" % (np.mean(qms), np.std(qms))

#         print(f'RI = {ri}\nF1 = {f1}\nPR AUC = {pr}\nROC AUC = {roc}')
        print(f'PR AUC: {pr_auc}')

        return {'PR AUC': pr_auc}#{'RI':ri,'F1':f1,'PR AUC':pr,'ROC AUC':roc}
