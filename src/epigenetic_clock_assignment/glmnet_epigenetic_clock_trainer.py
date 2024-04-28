import os
import logging
from dataclasses import dataclass
from typing import Optional, Union, Tuple
from functools import partial
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import numpy.typing as npt
import sigfig
from sklearn.model_selection import train_test_split
# from sklearn.metrics import median_absolute_error, r2_score
import scipy.stats as stats
import sklearn.metrics as metrics

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


@dataclass
class HyperParameterOptimizationResult:
    alpha: float
    lambd: float
    r2_mean: float
    r2_std: float


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
        verbose: bool = True
    ) -> Tuple[HyperParameterOptimizationResult, pd.DataFrame, ElasticNet]:

        alpha_range = np.linspace(0, 1, alpha_count)
        self._log.debug(
            'starting hyperparameter optimization with alpha range %s',
            alpha_range)

        models = []

        hyperparameter_stats = pd.DataFrame()

        for index, alpha in enumerate(alpha_range):
            self._log.debug('training with alpha %s...', alpha)

            model = self._create_model()(alpha=alpha, verbose=verbose)
            model.fit(X_train, y_train)

            train_stats = pd.DataFrame()
            train_stats['lambda'] = model.lambda_path_
            train_stats['alpha'] = alpha
            train_stats['cv_r2_mean'] = model.cv_mean_score_
            train_stats['cv_r2_std'] = model.cv_standard_error_
            train_stats['model_index'] = index

            hyperparameter_stats = pd.concat(
                (hyperparameter_stats, train_stats))

            models.append(model)

        # select best row based on r2_mean and self._std_error_weight_for_lambda_best * r2_std
        best_index = np.argmax(hyperparameter_stats['cv_r2_mean'] -
                               self._std_error_weight_for_lambda_best *
                               hyperparameter_stats['cv_r2_mean'])
        best_row = hyperparameter_stats.iloc[best_index]

        result = HyperParameterOptimizationResult(
            alpha=best_row['alpha'],
            lambd=best_row['lambda'],
            r2_mean=best_row['cv_r2_mean'],
            r2_std=best_row['cv_r2_std'])
        best_model = models[int(best_row['model_index'])]

        return result, hyperparameter_stats, best_model

    def calculate_statistics(self, y_true: Union[pd.Series, np.ndarray],
                             y_pred: Union[pd.Series, np.ndarray]):

        medae = metrics.median_absolute_error(y_true=y_true, y_pred=y_pred)
        r2 = metrics.r2_score(y_true=y_true, y_pred=y_pred)

        # linear regression statistics
        linregress_result = stats.linregress(x=y_true, y=y_pred)

    def plot_linear_regression_result(self, y_test: Union[pd.Series,
                                                          np.ndarray],
                                      y_pred: Union[pd.Series, np.ndarray]):

        sns.regplot(x=y_test, y=y_pred, ci=99, label='alma', n_boot=5000).set(
            title=
            f'Test set ($n={np.shape(X_test)[0]}$), $\\alpha={model.alpha:.2f}$, $\lambda={lamb:.2f}$, $R^2={r2:.2f}$, stderr={standard_error:.2f}, p={p_value} $MedAE={medae:.2f}$; \n(uncertainty bar for confidence interval of 95%)',
            xlabel='Age (years)',
            ylabel='DNAm age (years)')
        plt.show()

        pass

    def plot_hyperparameter_optimization_result(
            self, hyperparameter_result: HyperParameterOptimizationResult,
            hyperparameter_stats: pd.DataFrame):

        sns.set_theme()

        fig, ax = plt.figure(figsize=(15, 5)), plt.gca()

        sns.scatterplot(x=np.log10(hyperparameter_stats['lambda']),
                        y=hyperparameter_stats['alpha'],
                        hue=hyperparameter_stats['cv_r2_mean'],
                        size=hyperparameter_stats['cv_r2_std'],
                        sizes=(10, 250),
                        ax=ax,
                        legend='brief')
        ax.set_title(
            'Hyperparameter optimization result on training data\n' +
            'best is selected based on ' +
            f' $(CV\\ R^2\\ mean) - {self._std_error_weight_for_lambda_best}\\cdot(CV\\ R^2\\ std)$'
        )
        ax.set_xlabel('$log(\\lambda)$')
        ax.set_ylabel('$\\alpha$')
        L = ax.legend()
        L.get_texts()[0].set_text('$CV\\ R^2\\ mean$')
        L.get_texts()[6].set_text('$CV\\ R^2\\ std$')

        best_lambda_sigfig = sigfig.round(hyperparameter_result.lambd,
                                          sigfigs=4)
        best_alpha_sigfig = sigfig.round(hyperparameter_result.alpha,
                                         sigfigs=1)
        best_r2_mean_sigfig = sigfig.round(hyperparameter_result.r2_mean,
                                           sigfigs=4)
        best_r2_std_sigfig = sigfig.round(hyperparameter_result.r2_std,
                                          sigfigs=4)

        ax.annotate(
            f'$\lambda={best_lambda_sigfig},\\alpha={best_alpha_sigfig},CV\\ R^2\\ mean={best_r2_mean_sigfig}, CV\\ R^2\\ std={best_r2_std_sigfig}$',
            xy=(np.log10(hyperparameter_result.lambd),
                hyperparameter_result.alpha),
            xycoords='data',
            xytext=(-100, -30),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->',
                            connectionstyle='arc3',
                            color='red'))

        plt.show()

    def _create_model(self):
        return partial(
            ElasticNet,
            n_jobs=self._n_parallel_jobs,
            n_splits=self._n_cv_fold,
            cut_point=self._std_error_weight_for_lambda_best,
            random_state=self._seed,
        )
