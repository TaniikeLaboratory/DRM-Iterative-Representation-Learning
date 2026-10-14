import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

train_data = pd.read_csv("./data/DRM_loop14_descriptor_pool.csv", index_col=0)
test_data = pd.read_csv("./data/Test_example_C3_descriptor_pool.csv", index_col=0)
test_predicted = pd.read_csv("./data/Prediction_result_test_example.csv", index_col=0)
fileout = "./data/FPS_result_test_example.csv"
test_data = test_data.drop_duplicates()
test_predicted = test_predicted.drop_duplicates()

selected_descriptor_names = ["(mendeleev_number_max)^2","1/(bulk_modulus_max)","1/(lattice_constant_min)","1/ln(electron_negativity_max)","(atomic_volume_min)^3","exp(electron_negativity_ssd)","exp(gs_mag_moment_ase)","1/exp(covalent_radius_pyykko_triple_max)","(atomic_volume_max)^2","1/(c6_gb_min)","1/(covalent_radius_pyykko_triple_gmean)","1/exp(gs_mag_moment_ase)","1/ln(Polarizability_prod)","1/sqrt(gs_est_bcc_latcnt_sum)","1/sqrt(specific_heat_min)","ln(lattice_constant_sum)","ln(melting_point_gmean)","ln(vdw_radius_mm3_gmean)","sqrt(density_max)","sqrt(period_ase)"] # loop-16 descriptor set

number_of_samples=20
USE_FILTER = True  # True: class = 1 and 25 < predicted H2 yield < 50; False: no filtering

# Make a combined list of catalyst names
catalysts_in_train = train_data.index.tolist()
catalysts_in_test = test_data.index.tolist()
combined_catalyst_list = catalysts_in_train + catalysts_in_test

# Extract selected descriptors from train and test sets
train_catalyst_descriptors = train_data[selected_descriptor_names].values
test_catalyst_descriptors = test_data[selected_descriptor_names].values

# Combine train and test sets
combined_catalyst_descriptors = np.vstack((train_catalyst_descriptors, test_catalyst_descriptors))

# Standardize the descriptors
scaler = StandardScaler()
standardized_combined_descriptors = scaler.fit_transform(combined_catalyst_descriptors)

# Perform PCA on the standardized data
pca = PCA(n_components=3)
reduced_combined_descriptors = pca.fit_transform(standardized_combined_descriptors)

# Assuming 'train_selected_indices' are the indices of selected points from the train set
train_indices = np.arange(len(train_data))
test_indices = np.arange(len(train_data),len(train_data)+len(test_data))

selected_indices = train_indices.copy()
unselected_indices = test_indices.copy()
pooled_catalysts=[]
pooled_distances=[]
pooled_predicted_classes=[]
pooled_predicted_yields=[]

i=0
while i < number_of_samples:
    distances = cdist(reduced_combined_descriptors[unselected_indices], reduced_combined_descriptors[selected_indices], 'euclidean') #The shape of distances is unselected(row) x selected(column)
    min_distances = np.min(distances, axis=1)
    sampled_index=unselected_indices[np.argmax(min_distances)]
    sampled_distance=max(min_distances)
    selected_indices = np.append(selected_indices,sampled_index)
    unselected_indices = unselected_indices[unselected_indices != sampled_index]
    if combined_catalyst_list[sampled_index] in catalysts_in_train:
        print(combined_catalyst_list[sampled_index])
        continue
    if (not USE_FILTER or
        (test_predicted.loc[combined_catalyst_list[sampled_index], "H2 yield_avg_class_predicted"] == 1 and
        test_predicted.loc[combined_catalyst_list[sampled_index], "H2 yield_avg_predicted"] > 25 and
         test_predicted.loc[combined_catalyst_list[sampled_index], "H2 yield_avg_predicted"] < 50)):
        pooled_catalysts = np.append(pooled_catalysts, combined_catalyst_list[sampled_index])
        pooled_distances = np.append(pooled_distances, sampled_distance)
        pooled_predicted_classes = np.append(pooled_predicted_classes, test_predicted.loc[combined_catalyst_list[sampled_index], "H2 yield_avg_class_predicted"])
        pooled_predicted_yields = np.append(pooled_predicted_yields, test_predicted.loc[combined_catalyst_list[sampled_index], "H2 yield_avg_predicted"])
        i=i+1

# Create a DataFrame from the arrays and save
df = pd.DataFrame({
    'Catalyst': pooled_catalysts,
    'Predicted_class': pooled_predicted_classes,
    'Predicted_yield': pooled_predicted_yields,
    'Distance': pooled_distances
})
df['Predicted_class'] = df['Predicted_class'].astype(int)
df.to_csv(fileout, index=False)

