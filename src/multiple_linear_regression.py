# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


SL = 0.05


def backwardElimination(x, y, sl):
    numVars = x.shape[1]
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x


# Importing the dataset
dataset = pd.read_csv('../data/50_Startups.csv')

# Encoding categorical data with dummy columns
dataset = pd.get_dummies(data=dataset,
                         prefix='State',
                         prefix_sep='_',
                         columns=['State'],
                         sparse=False,
                         drop_first=True)

# initialize model input
feature_cols = list(dataset.columns)
feature_cols.remove('Profit')
X = dataset.loc[:, feature_cols].values
y = dataset.loc[:, 'Profit'].values.reshape(-1, 1)

# Avoiding the Dummy Variable Trap
# This was accomplished with drop_first param

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# building the optimal model using Backward Elimination
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

X_Modeled = backwardElimination(X_opt, y, SL)

print(f'Optimized model input:\n{X_Modeled}')
