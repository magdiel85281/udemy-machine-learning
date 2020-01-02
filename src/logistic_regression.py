# Logistic Regression

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Importing the dataset
dataset = pd.read_csv('../data/Social_Network_Ads.csv')
X = dataset.loc[:, ['Age', 'EstimatedSalary']].values.astype(float)
y = dataset.loc[:, 'Purchased'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(penalty='l2',
                                dual=False,
                                tol=0.0001,
                                C=1.0,
                                fit_intercept=True,
                                intercept_scaling=1,
                                class_weight=None,
                                random_state=0,
                                solver='lbfgs',
                                max_iter=100,
                                multi_class='auto',
                                verbose=0,
                                warm_start=False,
                                n_jobs=None)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{cm}')

# Visualising the Training set results
X_set, y_set = X_train, y_train

fig = plt.figure(figsize=(16, 8))
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1,
                               step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1,
                               step=0.01))
plt.contourf(X1, X2,
             classifier.predict(np.array([X1.ravel(),
                                X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,
             cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[(y_set == j).ravel(), 0],
                X_set[(y_set == j).ravel(), 1],
                c=np.array([ListedColormap(('red', 'green'))(i)]),
                label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
# plt.show()
plt.savefig('../img/logistic_regression_train.png')

# Visualising the Test set results
X_set, y_set = X_test, y_test

fig = plt.figure(figsize=(16, 8))
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1,
                               step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1,
                               step=0.01))
plt.contourf(X1, X2,
             classifier.predict(np.array([X1.ravel(),
                                X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,
             cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[(y_set == j).ravel(), 0],
                X_set[(y_set == j).ravel(), 1],
                c=np.array([ListedColormap(('red', 'green'))(i)]),
                label=j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
# plt.show()
plt.savefig('../img/logistic_regression_test.png')