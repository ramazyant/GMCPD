import collections
import numpy as np
from scipy.stats import skew, skewnorm
from tqdm import tqdm


def cps(period=100, N_tot=1000, add=0):
    t = [np.floor(np.random.normal(period, period//10))]
    for i in range(1, N_tot//period):
        t.append(t[i - 1] + np.floor(np.random.normal(period, period//10)))
    return np.array(t).astype(int)


def dataset1(period=100, N_tot=1000, coef1=0.6, coef2=-0.5, delta=1, mu=0, sigma=1., random_seed=None, N=1):
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    t_n = cps(period, N_tot)
    
    T = [0, 1]
    X = [0, 0]
    L = [0, 0]
    
    for i in range(2, N_tot):
        if i in t_n:
            N += delta
            mu += N / 16
            L += [1]
        else:
            L += [0]
        T += [i]
        X += [coef1 * X[i - 1] + coef2 * X[i - 2] + np.random.normal(mu, sigma, 1)[0]]
    # X = X[:t_n[-1]]
    # L = L[:t_n[-1]]
    return np.array(X).reshape(-1, 1), np.array(L)



def dataset2(period=100, N_tot=1000, coef1=0.6, coef2=-0.5, mu=0, sigma=1., random_seed=None, N=1):
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    t_n = cps(period, N_tot)
    
    T = [0, 1]
    X = [0, 0]
    L = [0, 0]
    
    for i in range(2, N_tot):
        if i in t_n:
            N += 1
            L += [1]
        else:
            L += [0]
        if N % 2 == 1:
            sigma = 1
        if N % 2 == 0:
            sigma = np.log(np.e + N / 4)
        T += [i]
        X += [coef1 * X[i - 1] + coef2 * X[i - 2] + np.random.normal(mu, sigma, 1)[0]]
    # X = X[:t_n[-1]]
    # L = L[:t_n[-1]]
    return np.array(X).reshape(-1, 1), np.array(L)


def get_cov_change(X, L):
    cps = []
    for i in range(len(L)):
        if L[i] == 1:
            cps.append(i)
    res = []
    for i in range(len(cps) - 1):
        res.append(np.cov(X[cps[i] : cps[i + 1]].T[0], X[cps[i] : cps[i + 1]].T[1])[0, 1])
    return res


def dataset3(period=100, N_tot=1000, mu=0, sigma=1., random_seed=None, N=1):
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    delta = 1 / (N_tot // period)
    flag = False
    
    t_n = cps(period, N_tot - period)
    
    T = [0, 1]
    X = [np.random.multivariate_normal(mean=[0, 0], cov=[[1, delta * N], [delta * N, 1]], size=1)[0], 
         np.random.multivariate_normal(mean=[0, 0], cov=[[1, delta * N], [delta * N, 1]], size=1)[0]]
    L = [0, 0]
    
    for i in range(2, N_tot):
        if i in t_n:
            N += 1
            L += [1]
            if flag:
                flag = False
            else:
                flag = True
        else:
            L += [0]
        
        if flag == False:
            cov = [[1, delta * N], [delta * N, 1]]
        else:
            cov = [[1, -delta * N], [-delta * N, 1]]
            
        T += [i]
        ax = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=1)[0]
        X += [ax]
    # X = X[:t_n[-1]]
    # L = L[:t_n[-1]]
    return np.array(X), np.array(L)


def gen_skewed_instance(N, period=100, mu=0, sigma=1):
    x = skewnorm(N).rvs(period)
    y = (x - np.std(x))/np.mean(x)
#     print(skewnorm.ppf(0.01, N), skewnorm.ppf(0.99, N))
    y = y[y > -20]
    y = y[y < 20]
    if len(y) == 0:
        y = gen_skewed_instance(N, period)
    if N % 2 == 1:
        y = -y
    return y


def get_skewness_change(X, L):
    cps = []
    for i in range(len(L)):
        if L[i] == 1:
            cps.append(i)
#     print(cps)
    res = []
    for i in range(len(cps) - 1):
        res.append(skew(X[cps[i] : cps[i + 1]])[0])
    return res


def dataset4(period=100, N_tot=1000, delta=1, mu=0, sigma=1., add=2, random_seed=None, N=1):
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    change_flag = True
    
    t_n = cps(period, N_tot, add)
    
    X = np.random.normal(mu, sigma, t_n[0])
    L = np.zeros(N_tot + add*period)
    L[t_n[0]] = 1
    
    for i in range(1, len(t_n)):
        if change_flag:
            change_flag = False
            N += delta
            X = np.concatenate((X, gen_skewed_instance(N, t_n[i]-t_n[i-1], mu, sigma)))
            L[t_n[i]] = 1
        else:
            L[t_n[i]] = 1
            change_flag = True
            X = np.concatenate((X, np.random.normal(mu, sigma, t_n[i]-t_n[i-1])))
    # X = X[:t_n[-1]]
    # L = L[:t_n[-1]]
    return np.array(X).reshape(-1, 1), np.array(L)


def gen_linear_trend(period=100, coef=1):
    trend = []
    for i in range(period):
        trend.append(i * coef)
    trend = np.array(trend)
    trend -= coef * period // 2
    return trend


def dataset5(period=100, N_tot=1000, coef0=0.01, coef_delta=0.001, mu=0, sigma=1., add=2, random_seed=None, N=-1):
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    t_n = cps(period, N_tot, add)
    
    X = X = gen_linear_trend(t_n[0], coef0) + np.random.normal(mu, sigma, t_n[0])
    coef0 *= N
    L = np.zeros(N_tot + add*period)
    L[t_n[0]] = 1
    
    for i in range(1, len(t_n)):
        X = np.concatenate((X, gen_linear_trend(t_n[i]-t_n[i-1], coef0) + np.random.normal(mu, sigma, t_n[i]-t_n[i-1])))
        L[t_n[i]] = 1
        if coef0 < 0:
            coef0 *= N
            coef0 += coef_delta
        else:
            coef0 *= N
    # X = X[:t_n[-1]]
    # L = L[:t_n[-1]]
    return np.array(X).reshape(-1, 1), np.array(L)