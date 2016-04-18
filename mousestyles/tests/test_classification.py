from __future__ import print_function, absolute_import, division

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from mousestyles.classification import report_model_performance
from mousestyles.data_utils import load_data_classification


def test_report_model_performance():
    labels, features = load_data_classification()
    yhat = report_model_performance(KNeighborsClassifier(),
                                    labels[:, 0],
                                    features)
    assert type(yhat) is np.ndarray
    assert yhat.shape == (1921,)
