import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
fn = r'C:\Users\DELL I5558\Desktop\Python\NSW-ER01-8.csv'
dataframe = pd.read_csv(fn)
dataset = dataframe.values
X = dataset[:, 0:23].astype(float)
Y = dataset[:, 23]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=21)

model_lin = SVC(kernel='linear', C=1, gamma=10)
model_rbf = SVC(kernel='rbf', C=1, gamma=10)

model_lin.fit(X_train, y_train)
accuracy_lin = model_lin.score(X_test, y_test)
print('Accuracy without MinMaxScaler (linear kernel): {}'.format(accuracy_lin))

model_rbf.fit(X_train, y_train)
accuracy_rbf = model_rbf.score(X_test, y_test)
print('Accuracy without MinMaxScaler (rbf kernel): {}'.format(accuracy_rbf))

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=21)


model_lin.fit(X_train, y_train)
accuracy_lin_std = model_lin.score(X_test, y_test)
print('Accuracy with MinMaxScaler (linear kernel): {}'.format(accuracy_lin_std))

model_rbf.fit(X_train, y_train)
accuracy_rbf_std = model_rbf.score(X_test, y_test)
print('Accuracy with MinMaxScaler (rbf kernel): {}'.format(accuracy_rbf_std))
