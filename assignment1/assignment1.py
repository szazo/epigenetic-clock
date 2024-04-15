import numpy as np
import numpy.typing as npt
from collections import defaultdict
from os import fsencode, replace
from typing import Hashable
from glmnet import ElasticNet
import time
import pandas as pd
from sklearn.model_selection import train_test_split

# # load meta
# meta_df = pd.read_csv('data/GSE40279_family.soft-MetaData.csv', delimiter='|')

# splitted = meta_df['sample_title'].str.split(' ', expand=True)
# meta_df['source_name'] = 'X' + splitted[2]

# print(meta_df)

class Assignment1:

    _meta_filepath: str
    _features_filepath: str

    _X: pd.DataFrame
    _y: npt.NDArray[np.float_]

    _model: ElasticNet

    def __init__(self, meta_filepath: str,
                 features_filepath: str):

        self._meta_filepath = meta_filepath
        self._features_filepath = features_filepath

    def train(self, test_size=0.2, train_test_split_seed=42, train_seed=42):

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(self._X,
                                                            self._y,
                                                            test_size=test_size,
                                                            random_state=train_test_split_seed)

        print('X', X_train.shape, X_test.shape)
        print('Y', y_train.shape, y_test.shape)
        
        self._model = ElasticNet(n_jobs = 20, n_splits = 10, random_state=train_seed, verbose=True)
        self._model.fit(X_train, y_train)

        print(self._model.__dir__())

        return self._model
        
    def load(self):

        meta_df = self._load_meta()
        features_df = self._load_features()

        # join based on the IDs (to match the order)
        joined_df = meta_df.join(features_df)
        y: npt.NDArray[np.float_] = np.array(joined_df['age'].astype(float).values)

        # drop the index and age fields from the joined
        joined_df.drop('age', inplace=True, axis=1)
        X = joined_df

        # sanity check based on https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279
        assert X.shape == (656, 473034)
        assert y.shape == (656, )

        self._X = X
        self._y = y
        
        return X, y

    def _load_meta(self) -> pd.DataFrame:
        meta_df = pd.read_csv(self._meta_filepath, delimiter='|')

        splitted = meta_df['sample_title'].str.split(' ', expand=True)
        meta_df['source_name'] = 'X' + splitted[2]

        meta_df = meta_df[['source_name', 'age (y)']]
        meta_df = meta_df.rename(columns={'age (y)': 'age'})
        
        # set it as index
        meta_df = meta_df.set_index('source_name')

        return meta_df

    def _load_features(self) -> pd.DataFrame:

        # load sample rows to create dtype mapping
        sample_feature_df = pd.read_csv(self._features_filepath, delimiter='\t', nrows=10, index_col=0)
        dtype_dict = self._map_column_types(sample_feature_df)

        # load the whole data
        feature_df = pd.read_csv(self._features_filepath, delimiter='\t', index_col=0, dtype=dtype_dict)

        # reset the index name (because it is the index for the CpG sites)
        feature_df.index.rename(None, inplace=True)

        # transpose
        feature_df = feature_df.transpose(copy=False)

        return feature_df

    def _map_column_types(self, sample_df: pd.DataFrame):
        print('processing...')
        print(sample_df.index.names)

        dtypes = sample_df.dtypes

        dtype_dict = {}

        for col in sample_df.columns:
            if str(dtypes[col]) == 'float64':
                dtype_dict[col] = 'float32'
            else:
                dtype_dict[col] = str(dtypes[col])

        return dtype_dict


# assignment1 = Assignment1(meta_filepath='data/GSE40279_family.soft-MetaData.csv',
#                           features_filepath='data/GSE40279_average_beta.txt')

# assignment1.load_meta()

# print('loading features...')
# t1 = time.perf_counter()
# feature_df = assignment1.load_features()
# t2 = time.perf_counter()
# print(f'elapsed: {(t2 - t1):.2f}')

# print(feature_df)
# print(feature_df.dtypes)

# sample_feature_df = pd.read_csv('data/GSE40279_average_beta.txt', delimiter='\t', nrows=100, index_col=0)
# print(sample_feature_df)
# dtype_dict = process_column_types(sample_feature_df)
# #print('dict', dtype_dict)


# # load the features with the specified types
# print('loading....')
# feature_df = pd.read_csv('data/GSE40279_average_beta.txt', delimiter='\t', dtype=dtype_dict)
# print('renaming index...')
# #feature_df.index.rename(None, inplace=True)

# print('transposing...')
# feature_df.transpose(copy=False)

# print('done')
# print('DTYPES')
# print(feature_df.dtypes)
# print(feature_df)
# #print(feature_df['ID_REF'])

# # # load features
# # dtypes = defaultdict(lambda: 'float32')
# # dtypes['ID_REF'] = 'str'

# # feature_df = pd.read_csv('data/GSE40279_average_beta.txt', delimiter='\t', nrows=1000, dtype=dtypes).T
# print('getting memory....')
# print(feature_df.info(verbose=False, memory_usage='deep'))

# t2 = time.perf_counter()
# print(f'elapsed: {(t2 - t1):.2f}')

# from dask import dataframe as dd

# df = dd.read_csv('data/GSE40279_average_beta.txt')
# print(df.head())
#print(df.columns)
# print(feature_df)

# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279

#model = ElasticNet()
