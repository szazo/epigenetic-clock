import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scipystats
from statannotations.Annotator import Annotator


class HealthyMDDSAPlot:

    def age_histogram(self, df: pd.DataFrame):
        fig, ax = plt.figure(figsize=(12, 10)), plt.gca()

        sns.histplot(ax=ax,
                     data=df,
                     x='age',
                     hue='Condition',
                     multiple='stack')
        ax.set_xlabel('Age', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title('Age distribution by condition', fontsize=18)
        plt.show()

    def boxplot(self, df: pd.DataFrame, title):
        fig, ax = plt.figure(figsize=(12, 10)), plt.gca()

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
        pairs = [('Healthy', 'MDD'), ('Healthy', 'SA'), ('MDD', 'SA')]
        annotator = Annotator(ax, pairs, **plotting_parameters)
        annotator.configure(text_format="simple",
                            test_short_name='Mann-Whitney-Wilcoxon')
        annotator.set_pvalues(
            [healthy_vs_mdd.pvalue, healthy_vs_sa.pvalue, mdd_vs_sa.pvalue])
        annotator.annotate()
        ax.set_title(title, fontsize=18)
        ax.set_ylabel('Age acceleration', fontsize=14)
        plt.show()
