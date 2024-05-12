import logging
from typing import Optional, Union
from collections import defaultdict
import os
import pandas as pd
import dask.dataframe as dd
from dask.dataframe.io.csv import read_csv as dask_read_csv
import numpy as np
import numpy.typing as npt


class Assignment2RRBSDataSource:

    _log: logging.Logger

    _meta_csv_filepath: str
    _features_csv_filepath: str
    _features_pickle_cache_filepath: str

    def __init__(self, meta_csv_filepath: str, features_csv_filepath: str,
                 features_pickle_cache_filepath: str):

        self._log = logging.getLogger(__class__.__name__)

        self._meta_csv_filepath = meta_csv_filepath
        self._features_csv_filepath = features_csv_filepath
        self._features_pickle_cache_filepath = features_pickle_cache_filepath

    def load(self,
             expected_instance_count=182,
             expected_feature_count=5201794):

        self._log.debug('load')

        meta_df = self._load_meta()
        features_df = self._load_features()

        # join based on the IDs (to be sure)
        self._log.debug('joining based on ID')
        joined_df = meta_df.join(features_df)
        self._log.debug('joined; shape=%s', joined_df.shape)

        self._log.debug('creating y')
        y: npt.NDArray[np.float_] = np.array(
            joined_df['Age (years)'].astype(float).values)

        # drop fields to create the X
        self._log.debug('creating X')
        X = joined_df.iloc[:, 3:]

        # assert X.shape == (expected_instance_count,
        #                    expected_feature_count), 'X shape mismatch'
        # assert y.shape == (expected_instance_count, ), 'y shape mismatch'

        self._log.debug('loaded')

        result_meta_df = joined_df[['Gender', 'Condition']]

        return X, y, result_meta_df

    def _load_meta(self) -> pd.DataFrame:
        meta_df = pd.read_csv(self._meta_csv_filepath, index_col=0)

        return meta_df

    def _load_features(self) -> pd.DataFrame:

        if self._features_pickle_cache_filepath is not None and os.path.isfile(
                self._features_pickle_cache_filepath):
            # load using cache
            self._log.debug('loading features from cache file "%s"...',
                            self._features_pickle_cache_filepath)
            df = pd.read_pickle(self._features_pickle_cache_filepath)
            self._log.debug('loaded; shape=%s', df.shape)
            return df

        df = self._load_features_from_csv_with_fillna()

        self._log.debug('saving features into cache file "%s"...',
                        self._features_pickle_cache_filepath)
        df.to_pickle(self._features_pickle_cache_filepath)
        self._log.debug('cache created')

        return df

    def _load_features_from_csv(self):
        # read floating point values as float32 and string 'Pos' field as pyarrow string
        dtypes = defaultdict(lambda: 'float32')
        dtypes['Pos'] = str(pd.StringDtype(storage='pyarrow'))

        # load using dask
        self._log.debug('loading "%s" using dask...',
                        self._features_csv_filepath)
        dask_df = dask_read_csv(self._features_csv_filepath, dtype=dtypes)

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

    def _load_features_from_csv_with_fillna(self):
        # read floating point values as float32 and string 'Pos' field as pyarrow string
        dtypes = defaultdict(lambda: 'float32')
        dtypes['Pos'] = str(pd.StringDtype(storage='pyarrow'))

        # load using dask
        self._log.debug('loading "%s" using dask...',
                        self._features_csv_filepath)
        dask_df = dask_read_csv(self._features_csv_filepath,
                                dtype=dtypes,
                                blocksize="228MB")
        print('PARTS', dask_df.npartitions)

        # set the 'Pos' field as index, after transpose it will be the column name
        self._log.debug('setting "Pos" field as index...')
        dask_df = dask_df.set_index('Pos')

        # fillna using row mean
        self._log.debug('fillna using row mean...')
        dask_df_filled = dask_df.map_partitions(self.fill_na_with_row_mean,
                                                meta=dask_df)
        self._log.debug('fillna finished...')

        # convert to pandas dataframe
        self._log.debug('converting to pandas dataframe...')
        df = dask_df_filled.compute()
        self._log.debug('converted; shape=%s', df.shape)

        self._log.debug('transposing...')
        df = df.T
        self._log.debug('transposed; shape=%s', df.shape)

        return df

    def fill_na_with_row_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(lambda row: row.fillna(row.mean()), axis=1)

    def fill_na_with_row_median(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(lambda row: row.fillna(row.median()), axis=1)

    def load_features2(self) -> pd.DataFrame:

        if self._features_pickle_cache_filepath is not None and os.path.isfile(
                self._features_pickle_cache_filepath):
            # load using cache
            self._log.debug('loading features from cache file "%s"...',
                            self._features_pickle_cache_filepath)
            df = pd.read_pickle(self._features_pickle_cache_filepath)
            self._log.debug('loaded; shape=%s', df.shape)
            return df

        df = self.load_features_from_csv2()

        self._log.debug('saving features into cache file "%s"...',
                        self._features_pickle_cache_filepath)
        df.to_pickle(self._features_pickle_cache_filepath)
        self._log.debug('cache created')

        return df

    def load_features_from_csv2(self):
        # read floating point values as float32 and string 'Pos' field as pyarrow string
        dtypes = defaultdict(lambda: 'float32')
        dtypes['Pos'] = str(pd.StringDtype(storage='pyarrow'))

        # load using dask
        self._log.debug('loading "%s" using dask...',
                        self._features_csv_filepath)

        dask_df = dask_read_csv(self._features_csv_filepath, dtype=dtypes)

        # set the 'Pos' field as index, after transpose it will be the column name
        self._log.debug('setting "Pos" field as index...')
        dask_df = dask_df.set_index('Pos')

        #return dask_df
        # convert to pandas dataframe
        self._log.debug('converting to pandas dataframe...')
        df = dask_df.compute()
        self._log.debug('converted; shape=%s', df.shape)

        # self._log.debug('transposing...')
        # df = df.T
        # self._log.debug('transposed; shape=%s', df.shape)

        return df

    def load_features_from_csv3(self):
        # read floating point values as float32 and string 'Pos' field as pyarrow string
        dtypes = defaultdict(lambda: 'float32')
        dtypes['Pos'] = str(pd.StringDtype(storage='pyarrow'))

        # load using dask
        self._log.debug('loading "%s" using dask...',
                        self._features_csv_filepath)

        column_names = pd.read_csv(self._features_csv_filepath,
                                   nrows=0).columns
        types_dict = {'Pos': str(pd.StringDtype(storage='pyarrow'))}
        float32_types_dict = {
            column_name: 'float32'
            for column_name in column_names if column_name not in types_dict
        }
        # types_dict.update(float32_types_dict)
        # print(types_dict)

        df = pd.read_csv(self._features_csv_filepath, dtype=dtypes)

        # # set the 'Pos' field as index, after transpose it will be the column name
        # self._log.debug('setting "Pos" field as index...')
        # dask_df = dask_df.set_index('Pos')

        # #return dask_df
        # # convert to pandas dataframe
        # self._log.debug('converting to pandas dataframe...')
        # df = dask_df.compute()
        self._log.debug('converted; shape=%s', df.shape)

        # self._log.debug('transposing...')
        # df = df.T
        # self._log.debug('transposed; shape=%s', df.shape)

        return df

    def load_features_from_csv4(self):
        # read floating point values as float32 and string 'Pos' field as pyarrow string
        # dtypes = defaultdict(lambda: 'float32')
        # dtypes['Pos'] = str(pd.StringDtype(storage='pyarrow'))

        # load using dask
        self._log.debug('loading "%s" using dask...',
                        self._features_csv_filepath)

        dask_df = dask_read_csv(self._features_csv_filepath)
        return dask_df
