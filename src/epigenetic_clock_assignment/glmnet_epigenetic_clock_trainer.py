import os
import logging
from dataclasses import dataclass
from typing import Optional, Union
from functools import partial
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import numpy.typing as npt
import sigfig
from sklearn.model_selection import train_test_split
import dask.dataframe as dd
import pandas as pd
from glmnet import ElasticNet


@dataclass
class ModelTrainResults:
    lambda_best: float
    lambda_best_nonzero_coef_count: int
    lambda_best_index: int
    alpha: float

    # informations about all lambdas
    lambda_path: npt.NDArray[np.float_]
    lambda_cv_score_mean: npt.NDArray[np.float_]
    lambda_cv_score_std: npt.NDArray[np.float_]


class GlmNetEpigeneticClockTrainer:

    _log: logging.Logger

    _n_parallel_jobs: int
    # number of cross validation folds
    _n_cv_fold: int
    # the weight of standard error used to determine lambda_best, 0 means use lambda_max
    _std_error_weight_for_lambda_best: float
    _seed: Optional[int] = None

    def __init__(self,
                 n_parallel_jobs: int,
                 n_cv_fold: int,
                 std_error_weight_for_lambda_best: float,
                 seed: Optional[int] = None):

        self._log = logging.getLogger(__class__.__name__)

        self._n_parallel_jobs = n_parallel_jobs
        self._n_cv_fold = n_cv_fold
        self._std_error_weight_for_lambda_best = std_error_weight_for_lambda_best
        self._seed = seed

    def train_test_split(self, X: Union[pd.DataFrame, np.ndarray],
                         y: Union[pd.Series, np.ndarray], test_size: float):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self._seed)

        return X_train, X_test, y_train, y_test

    def hyperparameter_optimization(
            self,
            X_train: Union[pd.DataFrame, np.ndarray],
            y_train: Union[pd.Series, np.ndarray],
            alpha_count: int,  # number of uniform alpha test between 0..1
            verbose: bool = True):

        alpha_range = np.linspace(0, 1, alpha_count)
        self._log.debug(
            'starting hyperparameter optimization with alpha range %s',
            alpha_range)

        models = []

        hyperparameter_stats = pd.DataFrame()

        for alpha in alpha_range:
            self._log.debug('training with alpha %s...', alpha)

            model = self._create_model()(alpha=alpha, verbose=verbose)
            model.fit(X_train, y_train)

            train_stats = pd.DataFrame()
            train_stats['lambda'] = model.lambda_path_
            train_stats['alpha'] = alpha
            train_stats['cv_r2_mean'] = model.cv_mean_score_
            train_stats['cv_r2_std'] = model.cv_standard_error_

            hyperparameter_stats = pd.concat(
                (hyperparameter_stats, train_stats))

            models.append(model)

        return hyperparameter_stats

    def plot_hypterparameter_optimization_result(
            self, hyperparameter_stats: pd.DataFrame):

        fig, ax = plt.figure(figsize=(15, 5)), plt.gca()
        # scatter_plot = ax.scatter(np.log10(hyperparameter_stats['lambda']),
        #            hyperparameter_stats['alpha'],
        #            c=hyperparameter_stats['cv_r2_mean'])
        # fig.colorbar(scatter_plot, label='$R^2$')
        # plt.show()

        best_index = np.argmax(hyperparameter_stats['cv_r2_mean'] -
                               self._std_error_weight_for_lambda_best *
                               hyperparameter_stats['cv_r2_mean'])
        best_row = hyperparameter_stats.iloc[best_index]
        print('BEST', best_row)

        sns.scatterplot(x=np.log10(hyperparameter_stats['lambda']),
                        y=hyperparameter_stats['alpha'],
                        hue=hyperparameter_stats['cv_r2_mean'],
                        size=hyperparameter_stats['cv_r2_std'],
                        sizes=(10, 250),
                        ax=ax)
        ax.set_title('Hyperparameter optimization result on training data')
        ax.set_xlabel('$log(\\lambda)$')
        ax.set_ylabel('$\\alpha$')
        # ax.legend(labels=['alma', 'korte'])
        L = ax.legend()
        L.get_texts()[0].set_text('$CV\ R^2\ mean$')
        L.get_texts()[6].set_text('$CV\ R^2\ std$')

        best_alpha = best_row['alpha']
        best_lambda = best_row['lambda']
        best_r2_mean = best_row['cv_r2_mean']
        best_r2_std = best_row['cv_r2_std']

        ax.annotate(
            f'$\lambda={sigfig.round(best_lambda, sigfigs=4)},\\alpha={best_alpha},CV R^2 mean={best_r2_mean}, CV R^2 std={best_r2_std}$',
            xy=(np.log10(best_lambda), best_alpha),
            xycoords='data',
            xytext=(10, -20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3'))

        # ax.annotate("Annotation",
        #     xy=(x1, y1), xycoords='data',
        #     xytext=(x2, y2), textcoords='offset points',
        #     )

        plt.show()

    #     sns.scatterplot(
    # data=tips, x="total_bill", y="tip", hue="size", size="size",
    # sizes=(20, 200), legend="full"


#)

    def _create_model(self):
        return partial(
            ElasticNet,
            n_jobs=self._n_parallel_jobs,
            n_splits=self._n_cv_fold,
            cut_point=self._std_error_weight_for_lambda_best,
            random_state=self._seed,
        )
