import pandas as pd
from sklearn.preprocessing import  StandardScaler

df_red = pd.read_csv('red-wine-dataset.csv')
df_white = pd.read_csv('white-wine-dataset.csv')

## data engineering

## normalization
scaler = StandardScaler()
scaler.fit()


