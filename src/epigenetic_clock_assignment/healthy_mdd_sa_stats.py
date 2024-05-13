import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as scipystats


class HealthyMDDSAStats:

    def prepare_stat_dataframe(self, meta_df: pd.DataFrame,
                               y: npt.NDArray[np.float_],
                               y_pred: npt.NDArray[np.float_],
                               stats: pd.DataFrame):

        meta_with_stats_df = meta_df.copy()
        meta_with_stats_df['age_acceleration'] = stats.age_acceleration
        meta_with_stats_df['delta_age'] = stats.delta_age
        meta_with_stats_df['age'] = y
        meta_with_stats_df['pred_age'] = y_pred

        def check_normality(x: pd.Series):
            res = scipystats.shapiro(x)
            return res.pvalue > 0.05

        is_age_normal_per_group = meta_with_stats_df.groupby(
            'Condition')['age'].agg(
                is_normal_p_value=lambda x: check_normality(x))
        is_age_acceleration_normal_per_group = meta_with_stats_df.groupby(
            'Condition')['age_acceleration'].agg(
                is_normal_p_value=lambda x: check_normality(x))

        return meta_with_stats_df, is_age_normal_per_group, is_age_acceleration_normal_per_group

    def control_for_imbalanced_age(self, df: pd.DataFrame,
                                   tolerance_year: float):

        condition_groupby = df.groupby('Condition')
        healthy_group = condition_groupby.get_group('Healthy')
        mdd_group = condition_groupby.get_group('MDD')
        sa_group = condition_groupby.get_group('SA')

        mdd_sa_merged = pd.merge_asof(mdd_group.rename(columns={
            'age': 'age_mdd'
        }).sort_values('age_mdd'),
                                      sa_group.rename(columns={
                                          'age': 'age_sa'
                                      }).sort_values('age_sa'),
                                      left_on='age_mdd',
                                      right_on='age_sa',
                                      suffixes=('_mdd', '_sa'),
                                      direction='nearest',
                                      tolerance=tolerance_year)

        mdd_sa_healthy_merged = pd.merge_asof(
            left=mdd_sa_merged.sort_values('age_sa'),
            right=healthy_group.rename(
                columns={
                    'age': 'age_healthy',
                    'delta_age': 'delta_age_healthy',
                    'age_acceleration': 'age_acceleration_healthy'
                }).sort_values('age_healthy'),
            left_on='age_sa',
            right_on='age_healthy',
            direction='nearest',
            tolerance=tolerance_year)

        # use only required columns
        mdd_sa_healthy_merged = mdd_sa_healthy_merged[[
            'age_healthy', 'age_mdd', 'age_sa', 'delta_age_healthy',
            'delta_age_mdd', 'delta_age_sa', 'age_acceleration_healthy',
            'age_acceleration_mdd', 'age_acceleration_sa'
        ]]
        #healthy_mdd_sa_merged = healthy_mdd_sa_merged.dropna()
        mdd_sa_healthy_merged = mdd_sa_healthy_merged.dropna()

        # merge back into one df
        controlled_dfs = []

        class_names = ['Healthy', 'MDD', 'SA']
        for i, cls in enumerate(['healthy', 'mdd', 'sa']):
            controlled_df = mdd_sa_healthy_merged[[
                f'age_{cls}', f'delta_age_{cls}', f'age_acceleration_{cls}'
            ]]
            controlled_df = controlled_df.assign(Condition=class_names[i])
            controlled_df = controlled_df.rename(
                columns={
                    f'age_{cls}': 'age',
                    f'delta_age_{cls}': 'delta_age',
                    f'age_acceleration_{cls}': 'age_acceleration'
                })
            controlled_dfs.append(controlled_df)

        merged_controlled_dfs = pd.concat(controlled_dfs)

        return merged_controlled_dfs
