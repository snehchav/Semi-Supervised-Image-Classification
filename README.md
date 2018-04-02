# Semi-Supervised-Image-Classification
The code written on the understanding of the paper: "Manifold Regularization: A Geometric Framework for Learning from Labeled and Unlabeled Examples"

1)	Dataset
 The dataset used was the CIFAR-10. The dataset can be accessed through the link given below:
https://www.cs.toronto.edu/~kriz/cifar.html

2) Feature Extraction 
A)	Feature_HOG.py : Extracts HOG feature descriptors for the CIFAR dataset
B)	Feature_intensityvalues.py : Extraction grayscale intensity values for the CIFAR dataset

3)	Main Code
A)LapRLS
   	Singleclass_Singleview_permutations_S.py : Implements a supervised classifier for binary classification single-view case.
   	Singleclass_Singleview_permutations_SS.py : Implements a Semi-supervised classifier for binary classification single-view case.
   	Singleclass_Multiview_permutations_S.py : Implements a supervised classifier for binary classification multi-view case.
   	Singleclass_Multiview_permutations_SS.py : Implements a Semi-supervised classifier for binary classification multi-view case.
   	Multiclass_Singleview_permutations_S.py : Implements a supervised classifier for multi class and single-view case.
   	Multiclass_Singleview_permutations_SS.py : Implements a Semi-supervised classifier for multi class and single-view case.
   	Multiclass_Multiview_permutations_S.py : Implements a supervised classifier for multi class and Multi-view case.
   	Multiclass_Multiview_permutations_SS.py : Implements a Semi-supervised classifier for multi class and Multi-view case.
The codes were run on Anaconda (Spyder) – Python 2.7.


B)	LapSVM
   	Laplacian Support Vector Machine Single-view single class – LAPSVM_SV_SC_SS.py
   	Laplacian Support Vector Machine Single-view multi-class – LAPSVM_SV_MC_SS.py
   	Laplacian Support Vector Machine Multi-view single class -  LAPSVM_MV_SC_SS.py
   	Laplacian Support Vector Machine Multi-view multi-class – LAPSVM_MV_MC_SS.py
For supervised learning set the gamma_I (gamma_I1, gamma_I2) parameters in all the scripts as zero
Python version 3.6 is used for implementing and running all the scripts. 
Install CVXOPT V 1.1.9 library that is compatible with the system and python version used for running the scripts.

