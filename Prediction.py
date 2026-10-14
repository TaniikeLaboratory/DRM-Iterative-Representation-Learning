import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
np.set_printoptions(linewidth= 800)

datatrain="./data/DRM_loop14_descriptor_pool.csv"
datatest="./data/DRM_loop14_descriptor_pool.csv" # Prediction for trainset
dataout="./data/Prediction_result_train.csv"  # Prediction for trainset
#datatest="./data/Test_example_C3_descriptor_pool.csv" # Prediction for testset
#dataout="./data/Prediction_result_test_example.csv" # Prediction for testset

df_train = pd.read_csv(datatrain, index_col=0)
df_test = pd.read_csv(datatest, index_col=0)

target="H2 yield_avg"
thr=2
target_class=[]
tclass=target+"_class"

#feature_class=["1/(electron_negativity_max)^2","1/(num_valance_sum)","1/exp(lattice_constant_min)","1/sqrt(icsd_volume_avg)","exp(bulk_modulus_min)","exp(num_d_valence_max)","ln(lattice_constant_min)","(icsd_volume_min)^3"] #loop0
#feature_class=["1/(covalent_radius_pyykko_triple_min)^3","1/sqrt(num_p_valence_max)","exp(atomic_volume_avg)","exp(lattice_constant_min)","exp(num_d_valence_max)","ln(en_pauling_var)","1/sqrt(bulk_modulus_min)","exp(covalent_radius_pyykko_triple_max)"] #loop1
#feature_class=["(covalent_radius_pyykko_triple_max)^2","(gs_est_bcc_latcnt_min)^3","1/exp(atomic_radius_sum)","1/sqrt(lattice_constant_min)","ln(num_d_valence_max)","sqrt(gs_volume_per_avg)","sqrt(melting_point_max)","1/ln(evaporation_heat_max)"] #loop2
#feature_class=["(gs_volume_per_avg)^3","1/(en_pauling_max)","1/(num_p_valence_max)^3","1/exp(en_ghosh_ase)","1/ln(gs_volume_per_min)","1/sqrt(atomic_radius_rahm_avg)","ln(lattice_constant_min)","1/exp(atomic_radius_rahm_avg)"] #loop3
#feature_class=["1/(lattice_constant_min)^3","1/(num_p_valence_max)^3","1/exp(bulk_modulus_max)","1/sqrt(covalent_radius_pyykko_triple_avg)","exp(lattice_constant_min)","ln(gs_volume_per_avg)","1/ln(gs_volume_per_ssd)","1/ln(covalent_radius_pyykko_double_ssd)"] #loop4
#feature_class=["1/(gs_est_bcc_latcnt_avg)^2","1/sqrt(lattice_constant_min)","exp(num_p_valence_ssd)","ln(num_d_valence_max)","sqrt(bulk_modulus_ssd)","sqrt(lattice_constant_min)","1/ln(c6_gb_ssd)","1/ln(gs_energy_ssd)"] #loop5
#feature_class=["1/(gs_volume_per_avg)","1/(num_p_unfilled_avg)^3","1/ln(vdw_radius_mm3_ssd)","1/sqrt(num_d_valence_max)","sqrt(lattice_constant_min)","1/sqrt(num_unfilled_min)","(en_pauling_ssd)^2","(num_d_valence_avg)^3"] #loop6
#feature_class=["(gs_volume_per_avg)^2","1/(en_pauling_ase)^3","1/(num_p_unfilled_max)^2","1/sqrt(lattice_constant_min)","1/sqrt(num_d_valence_max)","1/exp(atomic_number_min)","1/sqrt(melting_point_max)","(num_d_valence_avg)^2"] #loop7
#feature_class=["1/(bulk_modulus_max)^2","1/(gs_est_fcc_latcnt_min)^2","1/(vdw_radius_uff_min)^2","exp(gs_volume_per_avg)","num_p_valence_max","vdw_radius_min","1/ln(icsd_volume_avg)","(lattice_constant_min)^3"] #loop8
#feature_class=["1/(en_pauling_max)","1/(lattice_constant_min)^2","1/ln(mendeleev_number_max)","sqrt(lattice_constant_min)","vdw_radius_uff_min","1/ln(molar_volume_avg)","(heat_of_formation_max)^3","(atomic_volume_avg)^3"] #loop9
#feature_class=["(lattice_constant_min)^3","1/(atomic_volume_avg)^3","1/(covalent_radius_cordero_ssd)^3","1/(en_pauling_max)^2","num_p_valence_max","(molar_volume_avg)^2","(thermal_conductivity_max)^3","exp(gs_est_bcc_latcnt_min)"] #loop10
#feature_class=["1/(bulk_modulus_ssd)","1/(molar_volume_avg)","1/exp(num_p_valence_ssd)","1/sqrt(lattice_constant_min)","lattice_constant_min","(en_pauling_max)^3","1/ln(vdw_radius_mm3_ssd)","(dipole_polarizability_prod)^2"] #loop11
#feature_class=["(lattice_constant_min)^3","1/(covalent_radius_pyykko_triple_max)","1/exp(lattice_constant_min)","1/ln(num_d_valence_max)","1/sqrt(lattice_constant_min)","hhi_r_ssd","1/(atomic_radius_rahm_min)","1/(en_pauling_ase)"] #loop12
#feature_class=["(gs_mag_moment_avg)^3","1/(atomic_radius_min)^2","1/(en_pauling_max)","1/(icsd_volume_min)^3","1/(mendeleev_number_max)^3","sqrt(lattice_constant_min)","(gs_volume_per_avg)^2","sqrt(sound_velocity_prod)"] #loop-13 with 5 depth
feature_class=["(mendeleev_number_max)^2","1/(bulk_modulus_max)","1/(lattice_constant_min)","1/ln(electron_negativity_max)","(atomic_volume_min)^3","exp(electron_negativity_ssd)","exp(gs_mag_moment_ase)","1/exp(covalent_radius_pyykko_triple_max)"] #loop-14 with 5 depth

#feature_reg=["(atomic_number_var)^2","1/(bulk_modulus_sum)","exp(fusion_enthalpy_var)","sqrt(thermal_conductivity_ssd)","1/ln(density_gmean)","1/exp(hhi_p_max)","1/(en_pauling_max)^3","(bulk_modulus_max)^2"] #loop0
#feature_reg=["(thermal_conductivity_max)^2","1/(melting_point_max)^3","1/(num_d_unfilled_avg)","1/(sound_velocity_sum)","exp(vdw_radius_mm3_gmean)","gs_volume_per_sum","1/(en_pauling_max)^2","sqrt(hhi_p_min)"] #loop1
#feature_reg=["(covalent_radius_pyykko_min)^2","(thermal_conductivity_max)^3","sqrt(hhi_p_min)","1/(num_s_valence_sum)^3","Polarizability_sum","(lattice_constant_min)^3","exp(sound_velocity_sum)","exp(electron_negativity_max)"] #loop2
#feature_reg=["(density_gmean)^3","(thermal_conductivity_max)^3","1/ln(hhi_p_min)","vdw_radius_mm3_sum","(covalent_radius_pyykko_min)^2","1/exp(atomic_weight_avg)","ln(vdw_radius_alvarez_sum)","exp(en_pauling_max)"] #loop3
#feature_reg=["(density_max)^3","(thermal_conductivity_max)^3","1/(bulk_modulus_sum)^3","covalent_radius_pyykko_triple_avg","ln(lattice_constant_min)","ln(num_s_unfilled_sum)","(hhi_r_gmean)^3","1/(atomic_number_ase)"] #loop4
#feature_reg=["(thermal_conductivity_max)^2","1/(bulk_modulus_sum)^3","1/exp(hhi_r_prod)","ln(covalent_radius_pyykko_triple_avg)","ln(electron_affinity_ssd)","sqrt(num_s_valence_max)","exp(density_max)","1/(num_s_unfilled_ase)^3"] #loop5
#feature_reg=["(thermal_conductivity_max)^3","1/(bulk_modulus_sum)^3","sqrt(density_max)","ln(covalent_radius_slater_avg)","(period_max)^3","gs_mag_moment_ase","(density_prod)^2","mendeleev_number_prod"] #loop6
#feature_reg=["(num_s_unfilled_max)^3","(thermal_conductivity_max)^3","1/(bulk_modulus_sum)^3","1/(density_max)","1/(num_d_valence_ssd)^2","exp(num_s_valence_ssd)","ln(covalent_radius_pyykko_double_gmean)","num_f_valence_gmean"] #loop7
#feature_reg=["(hhi_p_max)^2","(num_f_valence_gmean)^2","1/(bulk_modulus_max)^3","1/(num_d_valence_max)^2","1/(num_d_valence_max)^3","1/sqrt(bulk_modulus_ssd)","ln(covalent_radius_cordero_avg)","melting_point_ase"] #loop8
#feature_reg=["(en_pauling_max)^2","(num_d_valence_max)^2","(num_f_valence_sum)^2","1/(density_max)","1/ln(num_unfilled_min)","1/sqrt(heat_capacity_mass_min)","covalent_radius_pyykko_triple_avg","num_d_valence_max"] #loop9
#feature_reg=["(atomic_weight_ase)^2","(en_pauling_max)^3","(num_d_valence_max)^3","1/(bulk_modulus_max)^3","1/(heat_of_formation_max)^3","1/(num_d_valence_ssd)^2","ln(covalent_radius_pyykko_double_gmean)","sqrt(bulk_modulus_max)"] #loop10
#feature_reg=["(atomic_number_ase)^2","1/(bulk_modulus_max)","1/(bulk_modulus_max)^3","1/(period_max)^3","1/(thermal_conductivity_sum)","1/exp(gs_mag_moment_ase)","1/ln(heat_capacity_molar_sum)","1/sqrt(en_pauling_ssd)","exp(gs_mag_moment_avg)","exp(num_d_valence_max)","ln(num_d_valence_max)","molar_volume_sum"] #loop-11 with 12 descriptors
#feature_reg=["(gs_mag_moment_max)^2","(sound_velocity_prod)^2","1/(c6_gb_gmean)","1/(covalent_radius_slater_gmean)","1/(gs_est_bcc_latcnt_max)^2","1/ln(first_ion_en_avg)","1/ln(heat_of_formation_gmean)","1/sqrt(lattice_constant_avg)","density_max","ln(vdw_radius_mm3_gmean)","sqrt(atomic_radius_rahm_min)","sqrt(heat_capacity_mass_min)"] #loop-12 with 12 descriptors
#feature_reg=["(bulk_modulus_max)^2","(gs_mag_moment_max)^2","(num_d_valence_max)^3","1/(bulk_modulus_max)^2","1/(bulk_modulus_ssd)","1/(lattice_constant_max)","1/(num_d_valence_ssd)^2","1/(period_max)^3","1/(sound_velocity_min)","1/exp(atomic_weight_sum)","1/ln(en_allen_max)","sqrt(covalent_radius_pyykko_double_gmean)"] #loop-13 with 12 descriptors
feature_reg=["(atomic_volume_max)^2","1/(c6_gb_min)","1/(covalent_radius_pyykko_triple_gmean)","1/exp(gs_mag_moment_ase)","1/ln(Polarizability_prod)","1/sqrt(gs_est_bcc_latcnt_sum)","1/sqrt(specific_heat_min)","ln(lattice_constant_sum)","ln(melting_point_gmean)","ln(vdw_radius_mm3_gmean)","sqrt(density_max)","sqrt(period_ase)"] #loop-14 with 12 descriptors


X_class_train=df_train[feature_class]
X_class_test=df_test[feature_class]
for i in range(len(df_train)):
    if (df_train.iloc[i,df_train.columns.get_loc(target)] >= thr):
        target_class.append(1)
    else:
        target_class.append(0)
df_train[tclass]=target_class
target_class=[]
for i in range(len(df_test)):
    if (df_test.iloc[i,df_test.columns.get_loc(target)] >= thr):
        target_class.append(1)
    else:
        target_class.append(0)
df_test[tclass]=target_class

y_class_train=df_train[tclass]

max_depth = 5  # 3 for Loop 0–12; 5 for Loop 13–14
LR=RandomForestClassifier(n_estimators=100,criterion="gini",min_samples_split=4,min_samples_leaf=2,
                           max_depth=max_depth,max_features="sqrt",class_weight="balanced")
LR.fit(X_class_train,y_class_train)
y_class_pred=LR.predict(X_class_train)
loo=LeaveOneOut()
print(metrics.balanced_accuracy_score(y_class_train, y_class_pred),np.mean(cross_val_score(LR, X_class_train, y_class_train, cv=loo, scoring="balanced_accuracy")))
y_class_test=LR.predict(X_class_test)
df_test[tclass+"_predicted"]=y_class_test

df_train=df_train[df_train[target] >= thr]
X_reg_train=df_train[feature_reg]
X_reg_test=df_test[feature_reg]
y_reg_train=df_train[target]
LR=HuberRegressor(max_iter=20000, epsilon=1.35)
LR.fit(X_reg_train,y_reg_train)
y_reg_pred=LR.predict(X_reg_train)
print(metrics.mean_absolute_error(y_reg_train,y_reg_pred),-np.mean(cross_val_score(LR, X_reg_train, y_reg_train, cv=loo, scoring="neg_mean_absolute_error")))
y_reg_test=LR.predict(X_reg_test)
df_test[target+"_predicted"]=y_reg_test

df_test.to_csv(dataout, columns=[tclass,target,tclass+"_predicted",target+"_predicted"], index=True)
