# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv('../data/Position_Salaries.csv')
X = dataset.loc[:, 'Level'].values.reshape(-1, 1)
y = dataset.loc[:, 'Salary'].values.reshape(-1, 1)

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Decision Tree Regression to the dataset
regressor = DecisionTreeRegressor(criterion='mse',
                                  splitter='best',
                                  max_depth=None,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  min_weight_fraction_leaf=0.0,
                                  max_features=None,
                                  random_state=0,
                                  max_leaf_nodes=None,
                                  min_impurity_decrease=0.0,
                                  min_impurity_split=None)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(np.array([[6.5]]))
print(f'Decision Tree Regressor Prediction: {y_pred}')

# Visualising the Decision Tree Regression results (higher resolution)
fig = plt.figure(figsize=(16, 8))
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
# plt.show()
plt.savefig('../img/decision_tree_regressor.png')