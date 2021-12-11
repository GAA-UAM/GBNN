""" cross-validation """

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

""" Cross-validation method """


def gridsearch(X, y, model, grid,
               scoring_functions=None, pipeline=None,
               best_scoring=True, random_state=None,
               n_cv_general=10, n_cv_intrain=10):

    cv_results_test = np.zeros((n_cv_general, 1))
    cv_results_generalization = np.zeros((n_cv_general, 1))
    pred = np.zeros_like(y)
    bestparams = []
    cv_results = []

    if type_of_target(y) == 'continuous':
        kfold_gen = KFold(n_splits=n_cv_general,
                          random_state=random_state, shuffle=True)

    elif type_of_target(y) == 'multiclass' or 'binary':
        kfold_gen = StratifiedKFold(n_splits=n_cv_general, shuffle=True,
                                    random_state=random_state)

    # k Fold cross-validation

    for cv_i, (train_index, test_index) in enumerate(kfold_gen.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if type_of_target(y) == 'continuous':
            kfold = KFold(n_splits=n_cv_intrain,
                          random_state=random_state, shuffle=True)

        elif type_of_target(y) == 'multiclass' or 'binary':
            kfold = StratifiedKFold(n_splits=n_cv_intrain, shuffle=True,
                                    random_state=random_state)

        estimator = model if pipeline is None else Pipeline(
            [pipeline, ('clf', model)])

        # Finding optimum hyper-parameter

        grid_search = GridSearchCV(estimator, grid, cv=kfold,
                                   scoring=scoring_functions,
                                   refit=best_scoring,
                                   return_train_score=False)

        grid_search.fit(x_train, y_train)

        pred[test_index] = grid_search.predict(x_test)

        bestparams.append(grid_search.best_params_)

        grid_search.cv_results_[
            'final_test_score'] = grid_search.score(x_test, y_test)

        cv_results.append(grid_search.cv_results_)

        cv_results_test[cv_i, 0] = grid_search.cv_results_[
            'mean_test_score'][grid_search.best_index_]
        cv_results_generalization[cv_i, 0] = grid_search.cv_results_[
            'final_test_score']

    results = {}
    results['Metric'] = [
        'Score' if scoring_functions is None else scoring_functions]
    results['Mean_test_score'] = np.mean(
        cv_results_test, axis=0)
    results['Std_test_score'] = np.std(
        cv_results_test, axis=0)
    results['Mean_generalization_score'] = np.mean(
        cv_results_generalization, axis=0)
    results['Std_generalization_score'] = np.std(
        cv_results_generalization, axis=0)

    pd.DataFrame(results).to_csv('Score.csv')
    pd.DataFrame(cv_results).to_csv('CV_results.csv')
    pd.DataFrame(bestparams).to_csv('Best_Parameters.csv')

    return results, pred
