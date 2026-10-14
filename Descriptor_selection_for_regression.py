import pandas as pd
import numpy as np
import copy
from joblib import parallel_backend
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from warnings import simplefilter
simplefilter(action='ignore')

# Loading data
df = pd.read_csv("./data/DRM_loop14_descriptor_pool.csv", index_col=0)
filename="./data/regression_descriptor_selection.csv"

# GA parameters
Npop=800
Elength=12 # 8 for Loops 0–10; 12 for Loops 11–14
Elmax=Elength-2
Ngen=1600
Nelt=Npop*2//10
Ncrs=Npop*6//10
Nmut=Npop*2//10

# Supervised ML and CV
LR = HuberRegressor(max_iter=500)
cv=LeaveOneOut()

# Setting X,y
target="H2 yield_avg"
thr=2.0
df = df[df[target] >= thr]
y = df[target]
droplist=["CH4 conv_avg","CH4 conv_se","CO2 conv_avg","CO2 conv_se","H2 yield_avg","H2 yield_se","CO yield_avg","CO yield_se","Conv balance_avg","Conv balance_se","C balance_avg","C balance_se","H balance_avg","H balance_se","O balance_avg","O balance_se"]
df=df.drop(columns=droplist)
feature_cols=df.columns.values

# Cutoff for offspring selection
MAEcut=-4
scl=1.5

# Save the best individual from each generation
def save_best(gen, combinations, scores):
    best = int(np.argmax(scores))
    row = pd.DataFrame([[gen, *combinations[best], -scores[best]]],
                       columns=["Gen"] + list(range(Elength)) + ["MAE"])
    row.to_csv(filename, mode="w" if gen == 0 else "a",
               header=(gen == 0), index=False)

# Lists for storing
Com0_arr=[]
MAE0_arr=[]

while len(Com0_arr) < Npop:
    Com=np.random.choice(feature_cols, Elength, replace=False)
    Com.sort()
    X=df[Com]
    with parallel_backend('loky', n_jobs=39):
        scores = cross_val_score(LR, X, y, cv=cv, scoring="neg_mean_absolute_error")
    MAE = np.mean(scores)
    if MAE < 0:
        Com0_arr.append(Com)
        MAE0_arr.append(MAE)

print("Gen,","MaxMAE,","AvgMAE", sep="")
print("0,",np.amax(MAE0_arr),",",np.average(MAE0_arr), sep="")
save_best(0, Com0_arr, MAE0_arr)

for i in range(1,Ngen):
    fit=np.exp(3*(MAE0_arr-np.amin(MAE0_arr))/(np.amax(MAE0_arr)-np.amin(MAE0_arr)))
    fit=fit/sum(fit)
    MAE1_arr=[]
    Com1_arr=[]
    A=np.array(MAE0_arr)
    k1=0
    while len(Com1_arr) < Nelt:
        where=np.argsort(A)[::-1]
        p1=where[k1]
        if all(len(np.intersect1d(Com0_arr[p1], Com_k2)) <= Elmax for Com_k2 in Com1_arr):
            Com1_arr.append(Com0_arr[p1])
            MAE1_arr.append(MAE0_arr[p1])
        k1=k1+1
    MAEcut=scl*np.min(MAE1_arr)

    while len(Com1_arr) < (Nelt+Ncrs):
        p1,p2=np.random.choice(range(Npop), 2, replace=False, p=fit)
        Common=np.intersect1d(Com0_arr[p1], Com0_arr[p2])
        if len(Common) == Elength:
            continue
        Diff=np.setxor1d(Com0_arr[p1], Com0_arr[p2])
        Com_sub=np.random.choice(Diff, Elength-len(Common), replace=False)
        Com=np.append(Common,Com_sub)
        Com.sort()
        X=df[Com]
        with parallel_backend('loky', n_jobs=39):
            scores = cross_val_score(LR, X, y, cv=cv, scoring="neg_mean_absolute_error")
        MAE = np.mean(scores)
        if MAE > MAEcut:
            Com1_arr.append(Com)
            MAE1_arr.append(MAE)

    while len(Com1_arr) < Nelt+Ncrs+Nmut:
        p1,p2=np.random.choice(range(Npop), 2, replace=False)
        Common=np.random.choice(Com0_arr[p1], Elength-1, replace=False)
        Com_sub=np.random.choice(feature_cols, 1)
        if len(np.intersect1d(Com0_arr[p1],Com_sub)) == 0:
            Com=np.append(Common,Com_sub)
            Com.sort()
            X=df[Com]
            with parallel_backend('loky', n_jobs=39):
                scores = cross_val_score(LR, X, y, cv=cv, scoring="neg_mean_absolute_error")
            MAE = np.mean(scores)
            if MAE > MAEcut:
                Com1_arr.append(Com)
                MAE1_arr.append(MAE)

    while len(Com1_arr) < Npop:
        Com=np.random.choice(feature_cols, Elength, replace=False)
        Com.sort()
        X=df[Com]
        with parallel_backend('loky', n_jobs=39):
            scores = cross_val_score(LR, X, y, cv=cv, scoring="neg_mean_absolute_error")
        MAE = np.mean(scores)
        if MAE > MAEcut:
            Com1_arr.append(Com)
            MAE1_arr.append(MAE)
            
    Com0_arr=copy.deepcopy(Com1_arr)
    MAE0_arr=copy.deepcopy(MAE1_arr)
    print(i,",",np.amax(MAE0_arr),",",np.average(MAE0_arr), sep="")
    save_best(i, Com0_arr, MAE0_arr)
