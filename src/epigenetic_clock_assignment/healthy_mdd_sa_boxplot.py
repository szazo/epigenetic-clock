import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scipystats
from statannotations.Annotator import Annotator


class HealthyMDDSABoxPlot:

    def age_histogram(self, df: pd.DataFrame):
        fig, ax = plt.figure(), plt.gca()

        sns.histplot(ax=ax,
                     data=df,
                     x='age',
                     hue='Condition',
                     multiple='stack')
        ax.set_xlabel('Age')
        ax.set_title('Sample distribution by age')
        plt.show()

    def boxplot(self, df: pd.DataFrame, title):
        fig, ax = plt.figure(), plt.gca()

        order = ['Healthy', 'MDD', 'SA']
        plotting_parameters = {
            'x': df['Condition'],
            'y': df['age_acceleration'],
            'order': order
        }

        sns.boxplot(ax=ax, **plotting_parameters)

        # comparison statistics
        condition_groupby = df.groupby('Condition')
        healthy_group = condition_groupby.get_group('Healthy')
        mdd_group = condition_groupby.get_group('MDD')
        sa_group = condition_groupby.get_group('SA')

        healthy_vs_mdd = scipystats.mannwhitneyu(
            healthy_group['age_acceleration'], mdd_group['age_acceleration'])
        healthy_vs_sa = scipystats.mannwhitneyu(
            healthy_group['age_acceleration'], sa_group['age_acceleration'])
        mdd_vs_sa = scipystats.mannwhitneyu(mdd_group['age_acceleration'],
                                            sa_group['age_acceleration'])
        # print('healthy_vs_mdd', healthy_vs_mdd)
        # print('healthy_vs_sa', healthy_vs_sa)
        # print('mdd_vs_sa', mdd_vs_sa)

        pairs = [('Healthy', 'MDD'), ('Healthy', 'SA'), ('MDD', 'SA')]
        annotator = Annotator(ax, pairs, **plotting_parameters)
        annotator.configure(text_format="simple",
                            test_short_name='Mann-Whitney-Wilcoxon')
        annotator.set_pvalues(
            [healthy_vs_mdd.pvalue, healthy_vs_sa.pvalue, mdd_vs_sa.pvalue])
        annotator.annotate()
        ax.set_title(title)
        ax.set_ylabel('Age acceleration')
        plt.show()
