from glmnet import ElasticNet
import pandas as pd

# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279

meta_df = pd.read_csv('data/GSE40279_family.soft-MetaData.csv', delimiter='|')
print(meta_df.head())

feature_df = pd.read_csv('data/GSE40279_average_beta.txt', delimiter='\t', index_col=0).T
print(feature_df.head())

model = ElasticNet()
