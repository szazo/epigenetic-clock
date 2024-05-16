import logging
import gc
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import numpy.typing as npt
import sigfig
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import sklearn.metrics as metrics

import pandas as pd
from glmnet import ElasticNet


@dataclass
class HyperParameterOptimizationResult:
    alpha: float
    lambd: float
    r2_mean: float
    r2_std: float
    nonzero_coefficient_count: int


@dataclass
class Statistics:
    r2: float
    slope: float
    intercept: float
    p_value: float
    standard_error: float
    medae: float
    delta_age: npt.NDArray[np.float_]
    age_acceleration: npt.NDArray[np.float_]


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

    def train_test_split(self, *arrays: List[Union[pd.DataFrame, np.ndarray]],
                         test_size: float):

        result = train_test_split(*arrays,
                                  test_size=test_size,
                                  random_state=self._seed)

        return result

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
            # paths used only for plotting
            train_stats['lambda_path'] = model.lambda_path_
            train_stats['cv_r2_mean_path'] = model.cv_mean_score_
            train_stats['cv_r2_std_path'] = model.cv_standard_error_
            # single parameters
            train_stats['lambda'] = model.lambda_best_[0]
            train_stats['alpha'] = alpha
            train_stats['cv_r2_mean'] = model.cv_mean_score_[
                model.lambda_best_inx_][0]
            train_stats['cv_r2_std'] = model.cv_standard_error_[
                model.lambda_best_inx_][0]
            train_stats['model_index'] = index
            train_stats[
                'nonzero_coefficient_count'] = self._count_nonzero_coefficients(
                    model)

            hyperparameter_stats = pd.concat(
                (hyperparameter_stats, train_stats))

            models.append(model)
            gc.collect()

        # select best row based on r2_mean and self._std_error_weight_for_lambda_best * r2_std
        # note: the first is enough, because the dataframe also contains lambda_path
        best_index = np.argmax(hyperparameter_stats['cv_r2_mean'] -
                               self._std_error_weight_for_lambda_best *
                               hyperparameter_stats['cv_r2_std'])
        best_row = hyperparameter_stats.iloc[best_index]

        result = HyperParameterOptimizationResult(
            alpha=best_row['alpha'],
            lambd=best_row['lambda'],
            r2_mean=best_row['cv_r2_mean'],
            r2_std=best_row['cv_r2_std'],
            nonzero_coefficient_count=best_row['nonzero_coefficient_count'])

        best_model = models[int(best_row['model_index'])]
        return result, hyperparameter_stats, best_model

    def calculate_statistics(
            self, y_true: Union[pd.Series, np.ndarray],
            y_pred: Union[pd.Series, np.ndarray]) -> Statistics:

        medae = metrics.median_absolute_error(y_true=y_true, y_pred=y_pred)

        # linear regression statistics
        slope, intercept, r, p, se = stats.linregress(x=y_true, y=y_pred)
        r2 = r**2

        # stat models
        # sm_y_true_with_constant = sm.add_constant(y_true)
        # sm_model = sm.OLS(y_pred, sm_y_true_with_constant).fit()
        # print(sm_model.summary())

        delta_age = y_pred - y_true
        age_acceleration = y_pred - (y_true * slope + intercept)

        result = Statistics(r2=r2,
                            slope=slope,
                            intercept=intercept,
                            p_value=p,
                            standard_error=se,
                            medae=medae,
                            delta_age=delta_age,
                            age_acceleration=age_acceleration)
        return result

    def _count_nonzero_coefficients(self, model: ElasticNet):

        coefs: npt.NDArray[np.float_] = np.array(model.coef_)
        atol = float(np.finfo(coefs.dtype).tiny)
        is_close_to_zero = np.isclose(coefs, 0, atol=atol, rtol=0)

        nonzero_count = np.count_nonzero(np.invert(is_close_to_zero))

        return nonzero_count

    def plot_linear_regression_result(
            self,
            y_true: Union[pd.Series, np.ndarray],
            y_pred: Union[pd.Series, np.ndarray],
            stats: Statistics,
            alpha: float,
            lamb: float,
            title_prefix: str,
            hue: Optional[Union[pd.Series, np.ndarray]] = None,
            style: Optional[Union[pd.Series, np.ndarray]] = None):

        fig, ax = plt.figure(figsize=(14, 8)), plt.gca()
        ax.set_aspect('equal', adjustable='box')

        alpha_sigfig = sigfig.round(alpha, sigfigs=1)
        lambda_sigfig = sigfig.round(lamb, sigfigs=3)
        r2_sigfig = sigfig.round(stats.r2, sigfigs=3)
        std_err_sigfig = sigfig.round(stats.standard_error, sigfigs=3)
        p_sigfig = sigfig.round(stats.p_value, sigfigs=2)
        medae_sigfig = sigfig.round(stats.medae, sigfigs=3)

        title = f'{title_prefix} ($n={np.shape(y_true)[0]}$); $\\alpha={alpha_sigfig}$, ' + \
            f' $\\lambda={lambda_sigfig}$\n$R^2={r2_sigfig}$, ' + \
            f'stderr={std_err_sigfig}, p={p_sigfig} ' + \
            f'$MedAE={medae_sigfig}$'

        sns.scatterplot(x=y_true, y=y_pred, ax=ax, hue=hue, style=style)

        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Age (years)', fontsize=16)
        ax.set_ylabel('DNAm age (years)', fontsize=16)

        regression_line_y = y_true * stats.slope + stats.intercept

        sns.lineplot(x=y_true,
                     y=regression_line_y,
                     ax=ax,
                     label='regression line')
        sns.lineplot(x=y_true, y=y_true, ax=ax, label='x=y')

        plt.show()

    def plot_hyperparameter_optimization_result(
        self,
        hyperparameter_result: HyperParameterOptimizationResult,
        hyperparameter_stats: pd.DataFrame,
        arrow_xytext_offset=(-100, -50)):

        sns.set_theme()

        fig, ax = plt.figure(figsize=(14, 8)), plt.gca()

        sns.scatterplot(x=np.log10(hyperparameter_stats['lambda_path']),
                        y=hyperparameter_stats['alpha'],
                        hue=hyperparameter_stats['cv_r2_mean_path'],
                        size=hyperparameter_stats['cv_r2_std_path'],
                        sizes=(10, 250),
                        ax=ax,
                        legend='brief')
        ax.set_title(
            'Hyperparameter optimization result on training data\n' +
            'best is selected based on ' +
            f' $(CV\\ R^2\\ mean) - {self._std_error_weight_for_lambda_best}\\cdot(CV\\ R^2\\ std)$',
            fontsize=18)
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
            f'$\\lambda={best_lambda_sigfig},\\alpha={best_alpha_sigfig},CV\\ R^2\\ mean={best_r2_mean_sigfig}, CV\\ R^2\\ std={best_r2_std_sigfig},n\\_coefs={hyperparameter_result.nonzero_coefficient_count}$',
            xy=(np.log10(hyperparameter_result.lambd),
                hyperparameter_result.alpha),
            xycoords='data',
            xytext=arrow_xytext_offset,
            textcoords='offset points',
            fontsize=15,
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
