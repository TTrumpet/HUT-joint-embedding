import pandas as pd
import numpy as np

name = 'musiccaps-public'

df = pd.read_csv(name + '.csv')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.8

train = df[msk]
test = df[~msk]

train.to_csv(name + 'train.csv', index=False)
test.to_csv(name + 'test.csv', index=False)