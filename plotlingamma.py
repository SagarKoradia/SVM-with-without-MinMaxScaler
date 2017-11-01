import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
fn = r'C:\Users\DELL I5558\Desktop\Python\ELEC5222\SVM\NSW-ER01.csv'
dataframe = pd.read_csv(fn)
dataset = dataframe.values
X = dataset[:, 0:23].astype(float)
Y = dataset[:, 23]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=21)

gamma = np.arange(1, 100)
train_accuracy = np.empty(len(gamma))
test_accuracy = np.empty(len(gamma))

for i, k in enumerate(gamma):
    model_lin = SVC(kernel='linear', C=1, gamma=k)

    model_lin.fit(X_train, y_train)
    accuracy_lin = model_lin.score(X_test, y_test)
    train_accuracy[i] = model_lin.score(X_train, y_train)
    test_accuracy[i] = model_lin.score(X_test, y_test)


plt.title('SVM: Varying Number of gamma')
plt.plot(gamma, test_accuracy, label='Testing Accuracy')
plt.plot(gamma, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Various value of gamma ')
plt.ylabel('Accuracy')
plt.show()
