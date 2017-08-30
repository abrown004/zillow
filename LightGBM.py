# Load packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split


# Load input files
features = pd.read_csv("../input/properties_2016.csv")
labels = pd.read_csv("../input/train_2016_v2.csv")


# Convert float64 data types to float32 to reduce memory
for c, dtype in zip(features.columns, features.dtypes):
    if dtype == np.float64:
        features[c] = features[c].astype(np.float32)
        
# Drop rows where all features are NaNs
features.dropna(axis = 'index', how = 'all')

# Exclude 99% quantile and 1% quantile from training data
print(len(labels))
outliers_high = labels["logerror"].quantile(0.97)
outliers_low = labels["logerror"].quantile(0.01)
labels = labels[labels["logerror"] < outliers_high]
labels = labels[labels["logerror"] > outliers_low]
print(len(labels))

# Join features to labels on ParcelID
df = labels.merge(features, how= 'left', on="parcelid")
