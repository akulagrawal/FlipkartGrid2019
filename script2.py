########### Create Train and Test csv files#############
import numpy as np
import pandas as pd
import shutil

df = pd.read_csv('./data/train_labels.csv')
split = 0.2
eval_len = int(0.2*len(df))
mask = np.random.randint(0, len(df), eval_len)

test_df = df.ix[mask]
train_df = df.drop(df.index[mask])

train_df.to_csv("./data/train_labels.csv", index=False)
test_df.to_csv("./data/test_labels.csv", index=False)

train_df.to_csv("./model/train_labels.csv", index=False)
test_df.to_csv("./model/test_labels.csv", index=False)
