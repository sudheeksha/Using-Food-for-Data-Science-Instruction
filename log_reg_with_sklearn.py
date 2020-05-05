import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("popcorn_v3.csv")

# Divide the data to training set and test set
data1 = data.head(30)
x = data1['Cook time(secs)']
y = data1['edible']

from sklearn.linear_model import LogisticRegression

# Create an instance and fit the model
lr_model = LogisticRegression(C = 0.01, solver='lbfgs', multi_class='ovr')
X = x.values.reshape(-1,1)
y = y.values.reshape(-1,1)
lr_model.fit(X, y)

prices = np.arange(1, 125, 1)
probabilities= []
for i in prices:
    p_loss, p_win = lr_model.predict_proba([[i]])[0]
    probabilities.append(p_win)

plt.scatter(data1['Cook time(secs)'], data1['edible'], label = 'cooked')

plt.plot(prices,probabilities, label = 'cooked')

data2 = data.head(35)
x = data2['Cook time(secs)']
y = data2['burnt']

lr_model2 = LogisticRegression(C = 0.01, solver='lbfgs', multi_class='ovr')

X = x.values.reshape(-1,1)
Y = y.values.reshape(-1,1)
lr_model2.fit(X, y)

plt.scatter(data2['Cook time(secs)'], data2['burnt'])
prices = np.arange(45, 150, 1)

probabilities = []
for i in prices:
    p_loss, p_win = lr_model2.predict_proba([[i]])[0]
    probabilities.append(p_win)

plt.plot(prices,probabilities, label = 'burnt')
plt.xlabel('Time in Seconds')
plt.ylabel('Class')
plt.legend()
plt.show()