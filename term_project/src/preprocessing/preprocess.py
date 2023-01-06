from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

# loading raw .arff data
data = loadarff('data/landing/raw_arff/speeddating.arff')
df = pd.DataFrame(data[0])
# backup to landing
df.to_csv('data/landing/raw_data.csv', index=False)


# ---*--- Basic Clean-up ---*---

# stripping b'' from string variables in raw DF
for col in df.columns:
    if (type(df[col][0]) == str) and (df[col][0][0]=='b'):
        df[col] = df[col].apply(lambda x: x[2:-1])

# assign rank to all ordinal variables in raw DF
for col in df.columns:
    # determine rank for all ordinal attributes except 'd_interests_correlate'
    if (type(df[col][0]) == str) and (df[col][0][0]=='[') and (col != 'd_interests_correlate'):
        range_list = [] # initialize empty range_list at every iteration
        # fill range_list
        for i in df[col]: 
            if i[1:-1].split('-') not in range_list:
                range_list.append(i[1:-1].split('-'))
        # convert ranges from str to int
        for i in range(0,len(range_list)):
            for j in range(0,len(range_list[i])):
                range_list[i][j] = int(range_list[i][j])

        # sort ranges ascending
        range_list.sort(key=lambda x: x[0])

        # convert ranges back from int to str
        for i in range(0,len(range_list)):
            for j in range(0,len(range_list[i])):
                range_list[i][j] = str(range_list[i][j])

        # map range to rank in sorted range_list
        def assign_rank(x, sorted_list=range_list):
            rank = sorted_list.index(x)
            return rank # maybe +1 ---> investigate!

        # apply lambda to replace str ranges with ordinal ranks
        df[col] = df[col].apply(lambda x: assign_rank(x[1:-1].split('-')))

# assign rank to 'd_interests_correlate'
df.d_interests_correlate.replace('[-1-0]', -1, inplace=True)
df.d_interests_correlate.replace('[0-0.33]', 0, inplace=True)
df.d_interests_correlate.replace('[0.33-1]', 1, inplace=True)


# ---*--- MICE Imputation Protocol ---*---

# 


# ---*--- Preprocessing Complete ---*---

# export clean-ish DF to landing
df.to_csv('data/formatted/clean_data.csv', index=False)