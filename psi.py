import numpy as np
import pandas as pd
from typing import Union, Tuple


def psi_calc(X1: Union[list, np.ndarray, pd.Series],
             X2: Union[list, np.ndarray, pd.Series],
             method_num_feature: Union[str, None] = "equidistant",
             bins: Union[int, list, None] = 10,
             force_categorical: bool = False,
             regulariser: float = 0.0001,
             precision: int = 3) -> Tuple[float, pd.DataFrame]:
    """
    Calculate PSI value for comparing two samples X1 and X2. Data will be binned
    (numerical) or assumed to already be categorical.

    The PSI value and frequency table are returned. If a bin/group is empty,
    its contribution to the PSI value in the calculation is controlled by
    the parameter 'regulariser'.

    Parameters
    ----------
    X1 : list, np.ndarray, pd.Series
        Sample 1 (expected values)
    X2 : list, np.ndarray, pd.Series
        Sample 2 (actual values)
    method_num_feature : str or None
        Method for binning, either 'equidistant' (default) or 'quantiles'.
        If it is None, the bins must be given explicitly.
    bins : int, list or None
        Number of bins for binning (int), or list of break points to use
        if it is a list. None only if it is categorical.
    force_categorical : bool
        Ignore that the input is numerical and treat it as categorical
    regulariser : float
        Frequency to assign empty bins/groups such that the PSI is well-defined
    precision : int
        number of digits to show in binning

    Returns
    -------
        Tuple
            - float: PSI-value
            - pd.DataFrame: frequency table
    """
    # initialise, perform checks and convert
    if isinstance(X1, pd.Series):
        pass
    elif (isinstance(X1, np.ndarray) and X1.ndim == 1) or isinstance(X1, list):
        X1 = pd.Series(X1)
    elif isinstance(X1, pd.DataFrame) and X1.shape[1] == 1:
        X1 = X1.squeeze()
    else:
        raise ValueError("Invalid type of input for X1")
    if isinstance(X2, pd.Series):
        pass
    elif (isinstance(X2, np.ndarray) and X2.ndim == 1) or isinstance(X2, list):
        X2 = pd.Series(X2)
    elif isinstance(X2, pd.DataFrame) and X2.shape[1] == 1:
        X2 = X2.squeeze()
    else:
        raise ValueError("Invalid type of input for X2")

    # if empty return
    if X1.empty or X2.empty:
        return np.nan, pd.DataFrame()

    # step 2: perform binning and calculate frequencies
    if (X1.dtype == "int" or X1.dtype == "float") and not force_categorical:  # considered to be numerical data
        assert X2.dtype == "int" or X2.dtype == "float", "X2 is not numerical while X1 is."

        # bin first X1 (the expected)
        if method_num_feature == "equidistant" or method_num_feature is None:
            if method_num_feature is None:
                assert isinstance(bins, list) or isinstance(bins, np.ndarray), \
                    "When method_num_feature, breaks must be given explicitly"
            X1, breaks = pd.cut(X1,
                                bins=bins,
                                retbins=True,
                                include_lowest=True,
                                precision=precision,
                                duplicates="drop")
            X1 = X1.astype(str)
        elif method_num_feature == "quantiles":
            assert isinstance(bins, int), "for method_num_feature=quantiles, bins must be an integer"
            X1, breaks = pd.qcut(X1,
                                 q=bins,
                                 retbins=True,
                                 precision=precision,
                                 duplicates="drop")
            X1 = X1.astype(str)
        else:
            raise ValueError("Invalid method chosen in method_num_feature")
        # apply expected binning to actual observations
        X2 = pd.cut(X2, bins=breaks, include_lowest=True, precision=precision).astype(str)

    else:  # considered to be categorical data
        pass
    X1f = X1.value_counts(dropna=False) / len(X1)
    X2f = X2.value_counts(dropna=False) / len(X2)
    X1f.name = "expected"
    X2f.name = "actual"

    # regularise empty bins/groups
    df = pd.DataFrame(X1f).join(X2f, how="outer")
    df0 = df.fillna(0.0).astype(float)  # the empty bins/groups are assigned zero frequency in the returned calculation
    df1 = df0.replace({0.0: regulariser}).astype(float)  # the empty bins/groups are regularised in the calculation

    # calculate PSI for each bin/group and then sum
    psi_value = df1.apply(lambda x: (x["expected"] - x["actual"]) * np.log(x["expected"] / x["actual"]), axis=1)
    psi_value = sum(psi_value)

    return psi_value, df0


# %%
if __name__ == "__main__2":
    # realistic example
    X1 = np.array([1, 2, 3, 3, 5, 4, 2, 2, 3, 2, 3, 4, 2, 2, 2, 5, 4, 5, 7, np.nan, 7])
    X2 = np.array([1, 2, 2, 3, 5, 4, 2, np.nan, 3, 2, np.nan, 4, 2, 2, 2, 5, 3, 5, 6, np.nan])

    psi_calc(X1=X1, X2=X2, method_num_feature="equidistant", bins=3)
    psi_calc(X1=X1, X2=X2, method_num_feature="quantiles", bins=3)
    psi_calc(X1=X1, X2=X2, force_categorical=True)

    # identical
    X1 = np.array([1, 2, 3, 3, 5, 4, 2, 2, 3, 2, 3, 4, 2, 2, 2, 5, 4, 5, 7, np.nan, 7])
    X2 = X1

    psi_calc(X1=X1, X2=X2, method_num_feature="equidistant", bins=3)

    # completely different
    X1 = [0.0, 0.1] * 100
    X2 = [1.0, 1.1] * 120

    psi_calc(X1=X1, X2=X2, method_num_feature="equidistant", bins=3)
