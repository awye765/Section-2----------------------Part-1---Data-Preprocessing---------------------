# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Stop numpy truncating arrays
np.set_printoptions(threshold=np.nan)

# Importing the dataset
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
# [:,] means we take all lines.  [, :-1] means we take all columns except for 
# last column, i.e. the "Purchased" column.  
# X is the FEATURES matrix.

y = dataset.iloc[:, 3].values
# [:, 3] means we take all lines and only the last column, i.e. the "Purchased"
# column.
# y are the DEPENDENT variables.

# Taking care of missing data
from sklearn.preprocessing import Imputer
# Imports from sklearn the Imputer class

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# Creates instance of Imputer class called "imputer" that takes THREE
# parameters.  First parameter specifies what type of missing value to
# look for, in this case any 'NaN'.  Second parameter specifies what strategy to
# use, in this case 'mean'.  Third parameter specifies which axis to use, i.e.
# the rows or columns, in this case we choose columns by setting access = 0.

imputer = imputer.fit(X[:, 1:3])
# Fits the above instance of the imputer class to X, i.e. the x variation of 
# the dataset. [:,] selects all rows.  [, 1:3] selects the columns at index 
# 1 and 2.  This way the imputer is limited in effect to the two columns that
# have missing data. 

X[:, 1:3] = imputer.transform(X[:, 1:3])
# Our imputer configuration is then used to transform the data in the columns
# that have missing data.  The result is that the columns that previously had 
# data now have data, i.e. salary and age data.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Imports from sklearn preprocessing library the LabelEncoder and 
# OneHotEncoder classes.

labelencoder_X = LabelEncoder()
# Creates instance of the LabelEncoder class, which we assign to the variable
# name labelencoder_X.

X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# Fitted labelencoder to first column, i.e. country column.  This is then
# assigned to the country column, updating that column for the X dataset.

onehotencoder = OneHotEncoder(categorical_features = [0])
# Creates instance of the OneHotEncoder class, with the parameter categorical_
# features set to index 0. This specifies the column (i.e. the column at 0 
# index) that we want to OneHotEncode.

X = onehotencoder.fit_transform(X).toarray()
# Transforms X by fitting onehotencoder object to array.  Don't need to specify
# 0 index as we've already specified in the onehotencoder object to target the
# column at index 0.

labelencoder_y = LabelEncoder()
# Creates instance of the LabelEncoder class.
y = labelencoder_y.fit_transform(y)
# Fits labelencoder object and transforms y object, which is the last column
# of our dataset, i.e. the "purchased column". 

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
# Imports from sklearn.cross_validation library the train_test_split library

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# X_train = training part of matrix of features. 
# X_test = training part of matrix of features.
# y_test takes an array of X values, array of y values.  0.2 means 20% of the
# dataset will be test data and 80% will be training.  Random_state = 0 keeps
# my results consistent with the instructors.
 

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
# Alot of ML models based on Euclidean distance, i.e. distance between two 
# points on a 2d graph represents difference between two features.  If one 
# feature has a different scale then the distance will be governed by that 
# feature.  Feature scaling normalises the data so each feature contributes
# proportionately to the final distance rather than disproportionately, as 
# would be the case without feature scaling.
