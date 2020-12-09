# SaSSaSSaS
#### Authors: Alice Townsend, Jean Merlet, Matt Lane, Luke Koch
### Files: 
- Notebooks
  - clustering_methods.ipynb 
  - fusion.ipynb
  - randomForestandBPNN.ipynb
  - mpp.ipynb 
  - fld.ipynb 
  - sota_cnn.ipynb
- Scripts
  - mpp_classifiers.py
  - load_ship_data.py 
  - load_flat_ncolor_data.py 

#### Instructions: 
The code that generated the results in our pdf is contained in the above files.
To verify our results, select the Jupyter Notebook whose name matches the algorithm results you wish to verify.
Any local .py files we imported are included as well.


#### Requirement

General steps involved in a pattern recognition problem include

- Data collection (raw data)
- Feature extraction (how to extract features from the raw data)
- Feature selection (dimensionality reduction - Fisher's linear discriminant or PCA)
- Classification/Regression methods need to be included
  - Supervised learning and Unsupervised learning
  - Baysian approaches and non-Baysian approaches
  - Parametric and Non-parametric density estimation in supervised learning
  - Fusion
- Performance evaluation
- Feedback system

You are required to evaluate the effect of various aspects of the classification/regression process, including but not limited to

- the effect of assuming the data is Gaussian-distributed
- the effect of assuming parametric pdf vs. non-parametric pdf
- the effect of using different prior probability ratio
- the effect of using different orders of Minkowski distance
- the effect of knowing the class label
- the effect of dimension of the feature space (e.g., changed through dimensionality reduction)
- the effect of fusion

To be more specific, you need to at least go through the following steps:

- Data normalization
- Dimensionality reduction
- Classification/Regression with the following
  - MPP (case 1, 2, and 3)
  - kNN with different k's
  - BPNN
  - Decision tree
  - SVM
  - Clustering (kmeans, wta, kohonen, or mean-shift)
- Classifier fusion
- Evaluation (use n-fold cross validation to generate confusion matrix and ROC curve if applicable).
