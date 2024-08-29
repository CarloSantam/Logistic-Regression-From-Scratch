
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,LinearRegression

np.random.seed(0)
X = np.random.normal(0, 1, 100).reshape(-1, 1)
y = (X > 0).astype(int).ravel()

model = LogisticRegression()
model.fit(X, y)

X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:, 1]

model2 = LinearRegression()
model2.fit(X, y)


X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
y_pred = model2.predict(X_test)


fig, axs = plt.subplots(1,figsize=(10, 5))


plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_prob, color='red', label='Logistic Funcion')
plt.legend()
plt.title('1D Logistic Function')
plt.show()


fig, axs = plt.subplots(1,figsize=(10, 5))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred, color='red', label='Linear Funcion')
plt.legend()
plt.title('1D Linear Regression on Binary Data')
plt.show()


def cross_entropy_loss(y_true, y_pred):
    #y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Dati di esempio
y_true = np.array(0)
y_pred = np.linspace(0.01, 0.99, 100)

loss = [cross_entropy_loss(y_true, p) for p in y_pred]

y_true = np.array(1)
y_pred = np.linspace(0.01, 0.99, 100)

loss_2 = [cross_entropy_loss(y_true, p) for p in y_pred]

fig, axs = plt.subplots(1,figsize=(10, 5))

plt.plot(y_pred, loss,label='y=0')
plt.xlabel('Predicted Probability')
plt.ylabel('Log-Likehood')
plt.title('Log-Likehood Loss vs Predicted')

plt.plot(y_pred, loss_2,label='y=1')
plt.xlabel('Predicted Probability')
plt.ylabel('Log-Likehood')
plt.title('Log-Likehood Loss vs Predicted')
plt.legend()

plt.show()


