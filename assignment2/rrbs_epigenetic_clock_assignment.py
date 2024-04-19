import os
import logging
from typing import Optional
from collections import defaultdict

import dask.dataframe as dd
import pandas as pd

class RRBSEpigeneticClockAssignment:

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

    def load(self):

        pass

    def _load_features(self):

        if self._features_pickle_cache_filepath is not None:
            
        

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
        
