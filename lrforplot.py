# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:26:03 2024

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Genera dati di esempio
np.random.seed(0)
X = np.random.normal(0, 1, 100).reshape(-1, 1)
y = (X > 0).astype(int).ravel()

# Crea e addestra il modello di regressione logistica
model = LogisticRegression()
model.fit(X, y)

# Genera punti per il grafico
X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:, 1]

# Crea il grafico
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_prob, color='red', label='Logistic Funcion')
plt.legend()
plt.title('1D Logistic Function')
plt.show()
