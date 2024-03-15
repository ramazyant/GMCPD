import torch
import ruptures
import numpy as np
import pandas as pd
from klcpd import KL_CPD
from TIRE import DenseTIRE as TIRE
from sklearn.preprocessing import StandardScaler
from density_ratio_estimator import DRChangeRateEstimator


def windows(X, width=50, model='rbf', jump=1):
    algo = ruptures.Window(width=2*width, model=model, jump=jump)
    algo.fit(X)
    
    scores = algo.score
    score = algo.score / width
    score = np.concatenate((np.zeros(width), score, np.zeros(width)))
    return score


def binseg(X, n_bkps = 50, model = 'rbf'):
    
    algo = ruptures.Binseg(model=model, jump=1).fit(X)
    my_bkps = algo.predict(n_bkps=n_bkps)[:-1] 
    score = np.zeros(len(X))
    score[my_bkps] = 1
    
    return score


def klcpd(X, window_size=50, n_epochs=1):
    ss = StandardScaler()
    X2 = ss.fit_transform(X)
    
    device = torch.device('cpu')
    model = KL_CPD(X.shape[1], p_wnd_dim=window_size, f_wnd_dim=window_size).to(device)
    model.fit(X2, epoches=n_epochs)

    score = model.predict(X2)
    return score


def tire(X, window_size=50, n_epochs=200):
    ss = StandardScaler()
    X2 = ss.fit_transform(X)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TIRE(input_dim=X.shape[1], window_size=window_size, 
                 intermediate_dim_TD=0, intermediate_dim_FD=25, nfft=50).to(device)

    model.fit(X2, epoches=n_epochs)

    dissimilarities, change_point_scores = model.predict(X2)
    score = np.concatenate((np.zeros(window_size), dissimilarities[1:], np.zeros(window_size)))
    return score


def rulsif(X, window_size=50):
    detector = DRChangeRateEstimator(sliding_window=X.shape[1], pside_len=window_size, cside_len=window_size,
                                     mergin=-1, trow_offset=0, tcol_offset=0)
    
    detector.build(estimation_method="RuLSIFitting", options=detector.RuLSIF_OPTION)

    score = detector.transform(X[:, :], destination="forward_backward")
    score = np.nan_to_num(score)
    return score