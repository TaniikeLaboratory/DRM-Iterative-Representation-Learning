import pandas as pd
import numpy as np
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from warnings import simplefilter
np.set_printoptions(linewidth= 600)
simplefilter(action='ignore')

df = pd.read_csv("./data/DRM_loop14_descriptor_pool.csv", index_col=0)
filename="./data/classification_descriptor_selection.csv"
Npop=100
Elength=8
Ngen=400

target="H2 yield_avg"
thr=2

target_class=[]
tclass=target+"_class"

for i in range(len(df)):
    if (df.iloc[i,df.columns.get_loc(target)] >= thr):
        target_class.append(1)
    else:
        target_class.append(0)

df[tclass]=target_class
y = df[tclass]

droplist=["CH4 conv_avg","CH4 conv_se","CO2 conv_avg","CO2 conv_se","H2 yield_avg","H2 yield_se","CO yield_avg","CO yield_se","Conv balance_avg","Conv balance_se","C balance_avg","C balance_se","H balance_avg","H balance_se","O balance_avg","O balance_se",tclass]
df=df.drop(columns=droplist)
feature_cols=df.columns.values


countlist=np.zeros_like(feature_cols)
df_count=pd.DataFrame(data=countlist,index=feature_cols, columns=["count"])
df_count=df_count.T
df_count.loc["count"] = +1

cv_method = LeaveOneOut()
cv_scoring="balanced_accuracy"
LR=RandomForestClassifier(n_estimators=100,criterion="gini",min_samples_split=4,min_samples_leaf=2,
                           max_depth=3,max_features="sqrt",class_weight="balanced")

dice=list(range(Npop))
Nelt0=Npop*2//10
Ncrs0=Npop*6//10
Nmut0=Npop*2//10
Nstr0=Npop-Nelt0-Ncrs0-Nmut0

MAEcut=-4
prc=0.02
Elmax=Elength-2
Pmax=2

scount=0
smax=10000

Com0_arr=[]
MAE0_arr=[]
MAE0_arr_max=-4

i=0
while i < Npop:
    Com=np.random.choice(feature_cols, Elength, replace=False)
    X=df[Com]
    scores = cross_val_score(LR, X, y, cv=cv_method, scoring=cv_scoring, n_jobs=32)
    MAE=np.mean(scores)
    if MAE > MAEcut:
        Com0_arr.append(Com)
        MAE0_arr.append(MAE)
        i=i+1

def save_best(gen, combinations, scores, mode="a", header=False):
    best = int(np.argmax(scores))
    record = pd.DataFrame([[gen, *combinations[best], scores[best]]],
                          columns=["Gen"] + list(range(Elength)) + ["Balanced_accuracy"])
    record.to_csv(filename, mode=mode, header=header, index=False)
    print(gen, scores[best], sep=",")

print("Gen,Best_accuracy")
save_best(0, Com0_arr, MAE0_arr, mode="w", header=True)

Nelt=Nelt0
Ncrs=Ncrs0
Nmut=Nmut0
Nstr=Nstr0

for i in range(1,Ngen):
    fit=np.exp(3*(MAE0_arr-np.amin(MAE0_arr))/(np.amax(MAE0_arr)-np.amin(MAE0_arr)))
    fit=fit/sum(fit)
    MAE1_arr=[]
    Com1_arr=[]
    A=np.array(MAE0_arr)
    j=0
    k1=0
    while j < Nelt:
        where=np.argsort(A)[::-1]
        p1=where[k1]
        Flag=0
        for k2 in range(len(Com1_arr)):
            if len(np.intersect1d(Com0_arr[p1], Com1_arr[k2])) > Elmax:
                Flag=1
        if Flag == 1:
            k1=k1+1
            continue
        Com1_arr.append(Com0_arr[p1])
        MAE1_arr.append(MAE0_arr[p1])
        j=j+1
        k1=k1+1
    if np.max(MAE1_arr) > MAE0_arr_max:
        Ncrs=Ncrs0
        Nmut=Nmut0
        Nstr=Nstr0
        scount=0
    else:
        scount=scount+1
        if scount>smax:
            Ncrs=Npop*2//10
            Nmut=Npop*5//10
            Nstr=Npop*1//10
    while j < (Nelt+Ncrs):
        p1,p2=np.random.choice(dice, 2, replace=False, p=fit)
        Common=np.intersect1d(Com0_arr[p1], Com0_arr[p2])
        if len(Common) == Elength:
            continue
        Diff=np.setxor1d(Com0_arr[p1], Com0_arr[p2])
        Com_sub=np.random.choice(Diff, Elength-len(Common), replace=False)
        Com=np.append(Common,Com_sub)
        X=df[Com]
        scores = cross_val_score(LR, X, y, cv=cv_method, scoring=cv_scoring, n_jobs=32)
        MAE=np.mean(scores)
        if MAE > MAEcut:
            df_count[Com]=df_count[Com]+(Pmax-df_count[Com])*prc
            Com1_arr.append(Com)
            MAE1_arr.append(MAE)
            j=j+1

        else:
            df_count[Com]=df_count[Com]-df_count[Com]*prc
    while j < Nelt+Ncrs+Nmut:
        p1,p2=np.random.choice(dice, 2, replace=False, p=fit)
        Common=np.random.choice(Com0_arr[p1], Elength-1, replace=False)
        prio = df_count.loc["count"].values.tolist()
        prio=prio/np.sum(prio)
        Com_sub=np.random.choice(feature_cols, 1, p=prio)
        if len(np.intersect1d(Com0_arr[p1],Com_sub)) == 0:
            Com=np.append(Common,Com_sub)
            X=df[Com]
            scores = cross_val_score(LR, X, y, cv=cv_method, scoring=cv_scoring, n_jobs=32)
            MAE=np.mean(scores)
            if MAE > MAEcut:
                df_count[Com]=df_count[Com]+(Pmax-df_count[Com])*prc
                Com1_arr.append(Com)
                MAE1_arr.append(MAE)
                j=j+1

            else:
                df_count[Com]=df_count[Com]-df_count[Com]*prc
    while j < Npop:
        prio = df_count.loc["count"].values.tolist()
        prio=prio/np.sum(prio)
        Com=np.random.choice(feature_cols, Elength, replace=False, p=prio)
        X=df[Com]
        scores = cross_val_score(LR, X, y, cv=cv_method, scoring=cv_scoring, n_jobs=32)
        MAE=np.mean(scores)
        if MAE > MAEcut:
            df_count[Com]=df_count[Com]+(Pmax-df_count[Com])*prc
            Com1_arr.append(Com)
            MAE1_arr.append(MAE)
            j=j+1
        else:
            df_count[Com]=df_count[Com]-df_count[Com]*prc
    MAE0_arr_max=np.max(MAE0_arr)
    Com0_arr=copy.deepcopy(Com1_arr)
    MAE0_arr=copy.deepcopy(MAE1_arr)
    save_best(i, Com0_arr, MAE0_arr)
