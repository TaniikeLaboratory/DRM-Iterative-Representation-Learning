# DRM-Iterative-Representation-Learning
Code, data, and reproducibility resources for active learning and iterative representation learning in DRM catalyst discovery.
- Generate_descriptor_pool.py: Generates catalyst descriptor pools from elemental compositions and normalized XenonPy properties through feature aggregation and nonlinear transformations. Supports both training and test datasets.
- Descriptor_selection_for_classification.py: Selects descriptor combinations using a genetic algorithm with a random forest classifier and leave-one-out cross-validation to classify catalysts by H2 yield (≥2% vs. <2%).
