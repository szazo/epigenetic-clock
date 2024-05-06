import logging
import pandas as pd
import numpy as np
import numpy.typing as npt


class Assignment1MicroarrayDataSource:

    _log: logging.Logger

    _meta_filepath: str
    _features_filepath: str

    def __init__(self, meta_filepath: str, features_filepath: str):

        self._log = logging.getLogger(__class__.__name__)

        self._meta_filepath = meta_filepath
        self._features_filepath = features_filepath

    def load(self, expected_instance_count=656, expected_feature_count=473034):

        self._log.debug('load')

        meta_df = self._load_meta()
        features_df = self._load_features()

        # join based on the IDs (to be sure)
        self._log.debug('joining based on ID')

        joined_df = meta_df.join(features_df)
        y: npt.NDArray[np.float_] = np.array(
            joined_df['age'].astype(float).values)

        # drop the age field from the joined
        joined_df.drop('age', inplace=True, axis=1)
        X = joined_df

        # sanity check based on https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279
        assert X.shape == (expected_instance_count, expected_feature_count)
        assert y.shape == (expected_instance_count, )

        self._log.debug('loaded')

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
        sample_feature_df = pd.read_csv(self._features_filepath,
                                        delimiter='\t',
                                        nrows=10,
                                        index_col=0)
        dtype_dict = self._map_column_types(sample_feature_df)

        # load the whole data
        feature_df = pd.read_csv(self._features_filepath,
                                 delimiter='\t',
                                 index_col=0,
                                 dtype=dtype_dict)

        # reset the index name (because it is the index for the CpG sites)
        feature_df.index.rename(None, inplace=True)

        # transpose
        feature_df = feature_df.transpose(copy=False)

        return feature_df

    def _map_column_types(self, sample_df: pd.DataFrame):
        dtypes = sample_df.dtypes

        dtype_dict = {}

        for col in sample_df.columns:
            if str(dtypes[col]) == 'float64':
                dtype_dict[col] = 'float32'
            else:
                dtype_dict[col] = str(dtypes[col])

        return dtype_dict
