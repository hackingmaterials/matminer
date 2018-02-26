import numpy as np
import pandas as pd

from pymatgen.core.structure import Structure

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, GridSearchCV
import sklearn.metrics
from sklearn.utils import shuffle

import matminer.featurizers.structure as MM
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.datasets.dataframe_loader import load_flla

import time

"""
The following script is an example of how to use matminer to run a kernel
ridge regression model on vector descriptors, in this case, coulomb
matrices (SCM) and orbital field matrices (OFM). The script follows 4 main steps:
1)  Retrieve the dataset. Since this script is attempting to learn
    formation energies from structure descriptors, formation energies
    and structures suffice for the dependent and independent variables.
    nsites and e_above_hull are retrieved for filtering the dataset
    and making the coulomb matrix eigenvalue lists the same size.
2)  Set up scikit-learn model. Five-fold cross validation is used with a
    four fold cross validation grid search for each training set.
    A parameter grid is selected to optimize the kernels on each
    training set before evaluating the model on the test set.
3)  Featurize the dataframe. This is done with the matminer multiprocessing
    option because featurization of large vector descriptors can be time
    consuming.
4)  Cross validation is run on the model, and mean average error (MAE),
    root mean square error (RMSE), and r-squared scores are calculated
    for each model.
"""

# If FABER is True, the script reads the list of material_ids in mpids.txt,
# which is the dataset used by Faber et al in their 2015 paper on the sine
# coulomb matrix. If FABER is False, all ternary oxides are retrieved from
# the Materials Project database, and the dataframe is filtered for stability
# and structure size.
FABER = True
FILTER = not FABER
NJOBS = 24
# Print parameters.
print("REMOVE UNSTABLE ENTRIES", FILTER)
print("USE FABER DATASET", FABER)
print("USE TERNARY OXIDE DATASET", not FABER)
print("NUMBER OF JOBS", NJOBS)

# Initialize data retrieval class
mpr = MPDataRetrieval()
# Choose query criteria
if FABER:
    df = load_flla()
else:
    criteria = "*-*-O"
    # Choose list of properties to retrive
    properties = ['structure', 'nsites', 'formation_energy_per_atom', 'e_above_hull']
    # Get the dataframe with the matching structure from the Materials Project
    df = mpr.get_dataframe(criteria=criteria, properties=properties)
    # Create the formation_energy feature for the SCM regression, since the SCM
    # model learns formation energy per unit cell rather than per atom.
    df['formation_energy'] = df['formation_energy_per_atom'] * df['nsites']
    # Structures are retrieved as dictionaries but can easily be converted to
    # pymatgen.core.Structure objects as shown.
    df['structure'] = pd.Series([Structure.from_dict(df['structure'][i])\
        for i in range(df.shape[0])], df.index)

# Filter the dataset if it consists of ternary oxides
if FILTER:
    df = df[df['e_above_hull'] < 0.1]
    df = df[df['nsites'] <= 30]
# Shuffle the structures to reduce bias
df = shuffle(df)
# Output dataframe size as debug info.
print("DF SHAPE", df.shape)
print()

# Hyperparameter space to search
# See the sklearn GridSearchCV documentation for details on parameter searching.
params = {}
params['sine coulomb matrix'] = [{'kernel' : ['rbf'], 'alpha' : [10**(-a) for a in range(2,6)],
    'gamma': [1/2.0/s/s for s in (20000,40000,80000,160000,320000)]},
    {'kernel' : ['laplacian'], 'alpha' : [10**(-a) for a in range(2,6)],
    'gamma' : [1.0/s for s in (20000,40000,80000,160000,320000)]}]
params['orbital field matrix'] = [{'alpha' : [10**(-a) for a in range(2,6)],
    'kernel': ['rbf'], 'gamma': [1/2.0/s/s for s in (2,4,8,16,32)]},
    {'alpha' : [10**(-a) for a in range(2,6)], 'kernel' : ['laplacian'],
    'gamma' : [1.0/s for s in (2,4,8,16,32)]}]

# Initialize the KFold cross validation splits used by the grid search algorithm.
inner_cv = KFold(n_splits=4, shuffle=False, random_state=0)

# Set up cross validation settings
nt = max(df['nsites'])
NUM_SPLITS = 5
inner_cv = KFold(n_splits=NUM_SPLITS-1, shuffle=False, random_state=0)
kf = KFold(NUM_SPLITS, False)

# Custom SCM KernelRidge estimator
# This estimator ensures that scores are based on formation energy per atom
# by dividing predicted and actual y values
# by the number of nonzero items in each row of the X matrix. This is equivalent
# to dividing by the number of sites in the corresponding structure
# because each vector descriptor is a list of eigenvalues of the SCM.
# The SCM is positive definite, so its eigenvalues are positive.
# This class only changes the results slightly, however, so the script can
# be simplified by replacing the SCM estimator below with a plain KernelRide()
# instance.
class KrrScm(KernelRidge):

    def __init__(self, alpha=1, kernel='linear', gamma = None, degree = 3, coef0 = 1, kernel_params = None):
        super(KrrScm, self).__init__(alpha, kernel, gamma, degree, coef0, kernel_params)

    def score(self, X, y):
        sizes = np.array([self.length(row) for row in X])
        y_pred = self.predict(X) / sizes
        y_true = y / sizes
        return sklearn.metrics.r2_score(y_true, y_pred)

    def length(self, vec):
        return vec[vec != 0].shape[0]

# SCM evaluation
DIAG = True
print ("DIAG ELEMS", DIAG)

# Featurize dataframe with sine coulomb matrix and time it
start = time.monotonic()
scm = MM.SineCoulombMatrix(DIAG)
# Set the number of jobs for parallelization
scm.set_n_jobs(NJOBS)
df = scm.featurize_dataframe(df, 'structure')
# Take the eigenvalues of the SCMs to form vector descriptors
df['sine coulomb matrix'] = pd.Series([np.sort(np.linalg.eigvals(s))[::-1] \
    for s in df['sine coulomb matrix']], df.index)
finish = time.monotonic()
print ("TIME TO FEATURIZE SCM %f SECONDS" % (finish-start))
print()

# Set up KRR model
krr = KrrScm()
print(krr.get_params().keys())
# Initialize hyperparameter grid search
hpsel = GridSearchCV(krr, params['sine coulomb matrix'], cv=inner_cv, refit=True)
X = df['sine coulomb matrix'].as_matrix()
# Append each vector descriptor with zeroes to make them all the same size.
XLIST = []
for i in range(len(X)):
    XLIST.append(np.append(X[i], np.zeros(nt - X[i].shape[0])))
X = np.array(XLIST)
print(X.shape)
Y = df['formation_energy'].as_matrix()
N = df['nsites'].as_matrix()
mae, rmse, r2 = 0, 0, 0
# Evaluate SCM and time it
start = time.monotonic()
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    N_train, N_test = N[train_index], N[test_index]
    hpsel.fit(X_train, Y_train)
    print("--- SCM PARAM OPT")
    print("---", hpsel.best_params_)
    Y_pred = hpsel.predict(X_test)
    mae += np.mean(np.abs(Y_pred - Y_test) / N_test) / NUM_SPLITS
    rmse += np.mean(((Y_pred - Y_test) / N_test)**2)**0.5 / NUM_SPLITS
    r2 += sklearn.metrics.r2_score(Y_test / N_test, Y_pred / N_test) / NUM_SPLITS
print ("SCM RESULTS MAE = %f, RMSE = %f, R-SQUARED = %f" % (mae, rmse, r2))
finish = time.monotonic()
print ("TIME TO TEST SCM %f SECONDS" % (finish-start))
print()

# OFM evaluation
for ROW in [False, True]:
    print ("ROW ELEMS", ROW)

    # Featurize dataframe with OFM and time it
    start = time.monotonic()
    ofm = MM.OrbitalFieldMatrix(ROW)
    ofm.set_n_jobs(NJOBS)
    df = ofm.featurize_dataframe(df, 'structure')
    df['orbital field matrix'] = pd.Series([s.flatten() \
        for s in df['orbital field matrix']], df.index)
    finish = time.monotonic()
    print ("TIME TO FEATURIZE OFM %f SECONDS" % (finish-start))
    print()

    # Get OFM descriptor and set up KRR model
    krr = KernelRidge()
    hpsel = GridSearchCV(krr, params['orbital field matrix'], cv=inner_cv, refit=True)
    X = df['orbital field matrix'].as_matrix()
    # Flatten each OFM to form a vector descriptor
    XLIST = []
    for i in range(len(X)):
        XLIST.append(X[i].flatten())
    X = np.array(XLIST)
    print(X.shape)
    Y = df['formation_energy_per_atom'].as_matrix()
    mae, rmse, r2 = 0, 0, 0
    # Evaluate OFM
    start = time.monotonic()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        hpsel.fit(X_train, Y_train)
        print("--- OFM PARAM OPT")
        print("---", hpsel.best_params_)
        Y_pred = hpsel.predict(X_test)
        mae += np.mean(np.abs(Y_pred - Y_test)) / NUM_SPLITS
        rmse += np.mean((Y_pred - Y_test)**2)**0.5 / NUM_SPLITS
        r2 += sklearn.metrics.r2_score(Y_test, Y_pred) / NUM_SPLITS
    print ("OFM RESULTS: MAE = %f, RMSE = %f, R-SQUARED = %f" % (mae, rmse, r2))
    finish = time.monotonic()
    print ("TIME TO TEST OFM %f SECONDS" % (finish-start))
    print()
    df.drop('orbital field matrix', 1, inplace = True)

"""

OUTPUT FOR FABER=True

REMOVE UNSTABLE ENTRIES False
USE FABER DATASET True
USE TERNARY OXIDE DATASET False
NUMBER OF JOBS 24
DF SHAPE (3938, 5)

DIAG ELEMS True
TIME TO FEATURIZE SCM 1.950617 SECONDS

dict_keys(['alpha', 'coef0', 'degree', 'gamma', 'kernel', 'kernel_params'])
(3938, 25)
--- SCM PARAM OPT
--- {'alpha': 0.001, 'gamma': 1.25e-05, 'kernel': 'laplacian'}
--- SCM PARAM OPT
--- {'alpha': 0.001, 'gamma': 1.25e-05, 'kernel': 'laplacian'}
--- SCM PARAM OPT
--- {'alpha': 0.001, 'gamma': 1.25e-05, 'kernel': 'laplacian'}
--- SCM PARAM OPT
--- {'alpha': 0.001, 'gamma': 1.25e-05, 'kernel': 'laplacian'}
--- SCM PARAM OPT
--- {'alpha': 0.001, 'gamma': 1.25e-05, 'kernel': 'laplacian'}
SCM RESULTS MAE = 0.386926, RMSE = 0.575041, R-SQUARED = 0.708389
TIME TO TEST SCM 461.997777 SECONDS

ROW ELEMS False
TIME TO FEATURIZE OFM 138.041182 SECONDS

(3938, 1024)
--- OFM PARAM OPT
--- {'alpha': 0.01, 'gamma': 0.125, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 0.01, 'gamma': 0.125, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 0.01, 'gamma': 0.125, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 0.01, 'gamma': 0.125, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 0.01, 'gamma': 0.125, 'kernel': 'laplacian'}
OFM RESULTS: MAE = 0.228587, RMSE = 0.346373, R-SQUARED = 0.894143
TIME TO TEST OFM 3039.900753 SECONDS

ROW ELEMS True
TIME TO FEATURIZE OFM 137.738271 SECONDS

(3938, 1521)
--- OFM PARAM OPT
--- {'alpha': 1e-05, 'gamma': 0.0625, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 1e-05, 'gamma': 0.0625, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 0.01, 'gamma': 0.125, 'kernel': 'rbf'}
--- OFM PARAM OPT
--- {'alpha': 0.01, 'gamma': 0.125, 'kernel': 'rbf'}
--- OFM PARAM OPT
--- {'alpha': 0.01, 'gamma': 0.125, 'kernel': 'rbf'}
OFM RESULTS: MAE = 0.171168, RMSE = 0.277463, R-SQUARED = 0.932034
TIME TO TEST OFM 4813.550195 SECONDS




OUTPUT FOR FABER=False

REMOVE UNSTABLE ENTRIES True
USE FABER DATASET False
USE TERNARY OXIDE DATASET True
DF SHAPE (7735, 5)

DIAG ELEMS True
TIME TO FEATURIZE SCM 5.249857 SECONDS

dict_keys(['alpha', 'coef0', 'degree', 'gamma', 'kernel', 'kernel_params'])
(7735, 30)
--- SCM PARAM OPT
--- {'alpha': 0.0001, 'gamma': 1.25e-05, 'kernel': 'laplacian'}
--- SCM PARAM OPT
--- {'alpha': 0.0001, 'gamma': 6.25e-06, 'kernel': 'laplacian'}
--- SCM PARAM OPT
--- {'alpha': 0.0001, 'gamma': 1.25e-05, 'kernel': 'laplacian'}
--- SCM PARAM OPT
--- {'alpha': 0.0001, 'gamma': 1.25e-05, 'kernel': 'laplacian'}
--- SCM PARAM OPT
--- {'alpha': 0.001, 'gamma': 1.25e-05, 'kernel': 'laplacian'}
SCM RESULTS MAE = 0.123372, RMSE = 0.220084, R-SQUARED = 0.916700
TIME TO TEST SCM 1815.762864 SECONDS

ROW ELEMS False
TIME TO FEATURIZE OFM 365.552943 SECONDS

(7735, 1024)
--- OFM PARAM OPT
--- {'alpha': 0.001, 'gamma': 0.125, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 0.01, 'gamma': 0.125, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 0.001, 'gamma': 0.125, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 0.01, 'gamma': 0.125, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 0.01, 'gamma': 0.125, 'kernel': 'laplacian'}
OFM RESULTS: MAE = 0.089616, RMSE = 0.139629, R-SQUARED = 0.966547
TIME TO TEST OFM 16216.300378 SECONDS

ROW ELEMS True
TIME TO FEATURIZE OFM 362.638039 SECONDS

(7735, 1521)
--- OFM PARAM OPT
--- {'alpha': 1e-05, 'gamma': 0.0625, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 1e-05, 'gamma': 0.0625, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 1e-05, 'gamma': 0.0625, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 1e-05, 'gamma': 0.0625, 'kernel': 'laplacian'}
--- OFM PARAM OPT
--- {'alpha': 1e-05, 'gamma': 0.0625, 'kernel': 'laplacian'}
OFM RESULTS: MAE = 0.058613, RMSE = 0.100224, R-SQUARED = 0.982667
TIME TO TEST OFM 25602.296397 SECONDS


"""
