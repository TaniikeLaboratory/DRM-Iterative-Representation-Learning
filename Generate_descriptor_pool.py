import numpy as np
import pandas as pd

# Data loading
df_descriptor = pd.read_csv("./data/xenonpy_normalized_v2.csv", index_col=0)
df_perform = pd.read_csv("./data/DRM_loop14.csv", index_col=0) # Training data
#df_perform = pd.read_csv("./data/Test_example_C3.csv", index_col=0) # Test data

desc_c = df_descriptor.columns.tolist()

target = df_perform[["CH4 conv_avg", "CH4 conv_se", "CO2 conv_avg", "CO2 conv_se", "H2 yield_avg", "H2 yield_se", "CO yield_avg","CO yield_se", "Conv balance_avg", "Conv balance_se", "C balance_avg", "C balance_se", "H balance_avg","H balance_se", "O balance_avg", "O balance_se"]]

#df_perform = df_perform[["Li","Al","Ca","V","Ni","Nb","Rh","Ag","Sn","Cs","Ba","Ce","Sm","Hf","Ir","Au","Bi"]] # E_0
#df_perform = df_perform[["Li","Mg","Al","K","Ca","Mn","Ni","Cu","Ga","Zr","Nb","Rh","Pd","Hf","W","Ir","Au"]] # E_1
#df_perform = df_perform[["Li","Al","Sc","Mn","Fe","Co","Ni","Zn","Y","Nb","Mo","Rh","Pd","Ta","W","Ir","Pt"]] # E_2
df_perform = df_perform[["Li","Mg","Al","K","Ca","Sc","V","Mn","Fe","Co","Ni","Cu","Zn","Ga","Y","Zr","Nb","Mo","Rh","Pd","Ag","Sn","Cs","Ba","Ce","Sm","Hf","Ta","W","Ir","Pt","Au","Bi"]] # E_all

perf_c = df_perform.columns.tolist()
perform = df_perform.values
df_de_extract = df_descriptor.loc[perf_c[:]].T
de_extract = df_de_extract.values
de = df_descriptor.T.values

result = []
column = []

for i in range(len(perform)):
    buffer1 = []
    buffer2 = []
    p_sum = sum(perform[i])
    for j in range(len(de_extract)):
        pro = perform[i] * de_extract[j]
        p_max = np.max(np.array(de_extract[j])[np.nonzero(np.array(perform[i]))])
        p_min = np.min(np.array(de_extract[j])[np.nonzero(np.array(perform[i]))])
        p_w_sum = sum(pro)
        p_w_avg = p_w_sum / p_sum
        p_ssd = sum((de_extract[j] - np.mean(de[j])) ** 2 * perform[i])
        p_w_asd = p_ssd / p_sum
        p_w_prod = np.prod(de_extract[j] ** perform[i])
        p_w_gmean = p_w_prod ** (1 / p_sum)
        buffer1.extend([p_min, p_max, p_w_sum, p_w_avg, p_ssd, p_w_asd, p_w_prod, p_w_gmean])
        buffer2.extend([desc_c[j] + "_min", desc_c[j] + "_max", desc_c[j] + "_sum", desc_c[j] + "_avg",
                        desc_c[j] + "_ssd", desc_c[j] + "_ase", desc_c[j] + "_prod", desc_c[j] + "_gmean"])
    result.append(buffer1)
    column.append(buffer2)

df_result = pd.DataFrame(result, columns=column[0], index=df_perform.index)

# Remove 0 variance and missing features
df_result = df_result.loc[:, (df_result != df_result.iloc[0]).any()]
df_result = df_result.dropna(axis=1)

# Identify descriptor columns before appending performance metrics
feature_cols0 = df_result.columns.tolist()
df = pd.concat([df_result, target], axis=1)

def Proto(ftype,feature,x_org):
    if ftype==0:
        Fx=feature
        x=x_org
    elif ftype==1:
        Fx="1/("+feature+")"
        x=1/x_org
    elif ftype==2:
        Fx="sqrt("+feature+")"
        x=x_org**0.5
    elif ftype==3:
        Fx="1/sqrt("+feature+")"
        x=1/(x_org**0.5)
    elif ftype==4:
        Fx="("+feature+")^2"
        x=x_org**2
    elif ftype==5:
        Fx="1/("+feature+")^2"
        x=1/(x_org**2)
    elif ftype==6:
        Fx="("+feature+")^3"
        x=x_org**3
    elif ftype==7:
        Fx="1/("+feature+")^3"
        x=1/(x_org**3)
    elif ftype==8:
        Fx="ln("+feature+")"
        x=np.log(x_org)
    elif ftype==9:
        Fx="1/ln("+feature+")"
        x=1/(np.log(x_org))
    elif ftype==10:
        Fx="exp("+feature+")"
        x=np.exp(x_org)
    elif ftype==11:
        Fx="1/exp("+feature+")"
        x=1/(np.exp(x_org))
    return(Fx,x)

#Generate non-linear featues
for i1 in range(len(feature_cols0)):
    x_temp1=df[feature_cols0[i1]]
    for j1 in range(12):
        Fx1,x1=Proto(j1,feature_cols0[i1],x_temp1)
        df[Fx1]=x1

#Save descriptor pool
df.to_csv("./data/DRM_loop14_descriptor_pool.csv")  # Training data
#df.to_csv("./data/Test_example_C3_descriptor_pool.csv")  # Test data




