import os
import logging
from typing import Optional, Union
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import dask.dataframe as dd
import pandas as pd
from glmnet import ElasticNet


class RRBSEpigeneticClockTrainer:

    _log: logging.Logger

    _meta_csv_filepath: str
    _features_csv_filepath: str
    _features_pickle_cache_filepath: str

    def __init__(self, meta_csv_filepath: str,
                 features_csv_filepath: str,
                 features_pickle_cache_filepath: str):

        self._meta_csv_filepath = meta_csv_filepath
        self._features_csv_filepath = features_csv_filepath
        self._features_pickle_cache_filepath = features_pickle_cache_filepath

        self._log = logging.getLogger(__class__.__name__)
        self._log.debug('__init__')

    def predict(self, X: Union[pd.DataFrame, np.ndarray], lamb: Optional[float] = None):

        assert self._model is not None, 'call train first'

        y_pred = self._model.predict(X, lamb=lamb)

        return y_pred
        
    def train(self,
              X_train: Union[pd.DataFrame, np.ndarray],
              y_train: np.ndarray,
              cv_fold: int = 3,
              parallel_jobs: int = 1,
              test_size=0.2,
              train_test_split_seed=42,
              train_seed=42):

        # train test split
        # X_train, X_test, y_train, y_test = train_test_split(self._X,
        #                                                     self._y,
        #                                                     test_size=test_size,
        #                                                     random_state=train_test_split_seed)

        
        self._model = ElasticNet(n_jobs = parallel_jobs,
                                 n_splits = cv_fold, random_state=train_seed, verbose=True)
        self._model.fit(X_train, y_train)

        return self._model

    def load(self):

        self._log.debug('load')

        meta_df = self._load_meta()
        features_df = self._load_features()

        # join based on the IDs (to be sure)
        self._log.debug('joining based on ID')
        features_df = meta_df.join(features_df)
        self._log.debug('joined; shape=%s', features_df.shape)

        self._log.debug('creating y')
        y: npt.NDArray[np.float_] = np.array(features_df['Age (years)'].astype(float).values)

        # drop fields to create the X
        self._log.debug('creating X')
        X = features_df.iloc[:, 3:]

        assert X.shape == (182, 5201794), 'X shape mismatch'
        assert y.shape == (182, ), 'y shape mismatch'

        self._log.debug('loaded')

        return X, y

    def _load_meta(self):
        meta_df = pd.read_csv(self._meta_csv_filepath, index_col=0)

        return meta_df

    def _load_features(self):

        if self._features_pickle_cache_filepath is not None and os.path.isfile(self._features_pickle_cache_filepath):
            # load using cache
            self._log.debug('loading features from cache file "%s"...', self._features_pickle_cache_filepath)
            df = pd.read_pickle(self._features_pickle_cache_filepath)
            self._log.debug('loaded; shape=%s', df.shape)
            return df

        df = self._load_features_from_csv()

        self._log.debug('saving features into cache file "%s"...', self._features_pickle_cache_filepath)
        df.to_pickle(self._features_pickle_cache_filepath)
        self._log.debug('cache created')

        return df

    def _load_features_from_csv(self):
        # read floating point values as float32 and string 'Pos' field as pyarrow string
        dtypes = defaultdict(lambda: 'float32')
        dtypes['Pos'] = str(pd.StringDtype(storage='pyarrow'))

        # load using dask
        self._log.debug('loading "%s" using dask...', self._features_csv_filepath)
        dask_df = dd.read_csv(self._features_csv_filepath, dtype=dtypes)

        # set the 'Pos' field as index, after transpose it will be the column name
        self._log.debug('setting "Pos" field as index...')
        dask_df = dask_df.set_index('Pos')

        # convert to pandas dataframe
        self._log.debug('converting to pandas dataframe...')
        df = dask_df.compute()
        self._log.debug('converted; shape=%s', df.shape)

        self._log.debug('transposing...')
        df = df.T
        self._log.debug('transposed; shape=%s', df.shape)

        return df
