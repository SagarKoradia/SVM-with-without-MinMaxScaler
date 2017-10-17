import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
fn = r'C:\Users\DELL I5558\Desktop\Python\NSW1.csv'
dataframe = pd.read_csv(fn)
dataset = dataframe.values
X = dataset[:, 0:20].astype(float)
Y = dataset[:, 20]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=21)

model_lin = SVC(kernel='linear', C=1, gamma=10)
model_rbf = SVC(kernel='rbf', C=1, gamma=10)

model_lin.fit(X_train, y_train)
accuracy = model_lin.score(X_test, y_test)
print(accuracy)

model_rbf.fit(X_train, y_train)
accuracy = model_rbf.score(X_test, y_test)
print(accuracy)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=21)


model_lin.fit(X_train, y_train)
accuracy = model_lin.score(X_test, y_test)
print(accuracy)

model_rbf.fit(X_train, y_train)
accuracy = model_rbf.score(X_test, y_test)
print(accuracy)
