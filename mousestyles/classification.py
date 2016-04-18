from __future__ import print_function, absolute_import, division

import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score

from mousestyles.data_utils import load_data_classification

seed = 1
model_list = [KNeighborsClassifier,
              LogisticRegression,
              GaussianNB,
              RandomForestClassifier,
              ExtraTreesClassifier,
              GradientBoostingClassifier,
              LinearSVC,
              SVC,
              LinearDiscriminantAnalysis,
              QuadraticDiscriminantAnalysis]

models_params = {
    "KNeighborsClassifier": {
        "n_neighbors": 5,
        "weights": 'uniform',
        "n_jobs": 1,
    },
    "LogisticRegression": {
        "solver": "lbfgs",
        "multi_class": "multinomial",
        "penalty": "l2",
        "C": 1.0,
    },
    "GaussianNB": {
    },
    "RandomForestClassifier": {
        "n_estimators": 500,
        "criterion": "gini",
        "max_depth": 8,
        "bootstrap": True,
        "random_state": seed,
        "verbose": 0,
        "n_jobs": -1,
    },
    "ExtraTreesClassifier": {
        "n_estimators": 500,
        "criterion": "gini",
        "max_depth": 8,
        "bootstrap": True,
        "random_state": seed,
        "verbose": 0,
        "n_jobs": -1,
    },
    "GradientBoostingClassifier": {
        "loss": "deviance",
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 1.0,
        "max_depth": 6,
        "random_state": seed,
        "max_features": 10,
        "verbose": 0,
    },
    "LinearSVC": {
        "penalty": "l2",
        "loss": "hinge",
        "C": 1.0,
        "verbose": 0,
        "random_state": seed,
        "multi_class": "ovr",
    },
    "SVC": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": .01,
        "random_state": seed,
        "verbose": 0,
    },
    "LinearDiscriminantAnalysis": {
        "solver": "lsqr",
        "shrinkage": "auto",
    },
    "QuadraticDiscriminantAnalysis": {
        "reg_param": .1,
    },
}


def report_model_performance(model, label, features):
    """ Compute CV accuracy for classifying mouse strain

    Parameters:
    -----------
    model: scikit-learn model
        A scikit-learn model, e.g. LogisticRegression()

    label: ndarray shape (1921,)
        array contains the mouse strain id from 0 to 15

    features: ndarray shape (1921, 99)
        array contain the mouse activity features

    Returns:
    --------
    yhat: ndarray shape (1921,):
        the predicted strain_id by cross validation

    This function also print out the cv accuracy.

    """
    # Set the model hyperparameter according to models_params
    start_time = time.time()
    hyper_params = models_params[model.__class__.__name__]
    # If the model already uses all CPU cores, Cross-Validation will not run in
    # parallel. If the model only use 1 core, Cross-Validation will run in
    # parallel.
    if "n_jobs" in hyper_params.keys():
        n_cv_jobs = (-1) * hyper_params["n_jobs"]
    else:
        n_cv_jobs = -1
    model.set_params(**models_params[model.__class__.__name__])
    cv = StratifiedKFold(label, n_folds=5, shuffle=True, random_state=seed)
    yhat_cv = cross_val_predict(model, features, label, cv=cv,
                                n_jobs=n_cv_jobs, verbose=0)
    time_spent = time.time() - start_time
    print("%30s CV Acc: %.4f in %6.2f seconds" % (
        model.__class__.__name__, accuracy_score(label, yhat_cv), time_spent))
    return yhat_cv


def main():
    """ Loop through models and report CV accuracy
    """
    labels, features = load_data_classification()
    for model in model_list:
        report_model_performance(model(), labels[:, 0], features)

if __name__ == "__main__":
    main()
