from psi import psi_calc
import numpy as np

# 1) Realistic example: different integers
X1 = np.array([1, 2, 3, 3, 5, 4, 2, 2, 3, 2, 3, 4, 2, 2, 2, 5, 4, 5, 7, np.nan, 7])
X2 = np.array([1, 2, 2, 3, 5, 4, 2, np.nan, 3, 2, np.nan, 4, 2, 2, 2, 5, 3, 5, 6, np.nan])

# consider data numerical
psi_calc(X1=X1, X2=X2, method_num_feature="equidistant", bins=3)
psi_calc(X1=X1, X2=X2, method_num_feature="quantiles", bins=3)
# consider data categorical
psi_calc(X1=X1, X2=X2, force_categorical=True)

# 2) Example: identical data
X1 = np.array([1, 2, 3, 3, 5, 4, 2, 2, 3, 2, 3, 4, 2, 2, 2, 5, 4, 5, 7, np.nan, 7])
X2 = X1
# consider data numerical
psi_calc(X1=X1, X2=X2, method_num_feature="equidistant", bins=3)

# 3) Example: completely different
X1 = [0.0, 0.1] * 100
X2 = [1.0, 1.1] * 120
# consider data numerical
psi_calc(X1=X1, X2=X2, method_num_feature="equidistant", bins=3)
