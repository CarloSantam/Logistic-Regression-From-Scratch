from numpy import log, dot
from sklearn.metrics import f1_score
from numpy.random import rand

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
import time
from mlxtend.plotting import plot_confusion_matrix


X = load_breast_cancer()['data'] 
y = load_breast_cancer()['target']
feature_names = load_breast_cancer()['feature_names'] 

scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

mi=mutual_info_classif(X_train, y_train, discrete_features='auto',random_state=42)

sorted_indices = np.argsort(mi)[::-1]

Feature_selected=20

sorted_indices=sorted_indices[:Feature_selected]

X_train=X_train[:,sorted_indices]

X_test=X_test[:,sorted_indices]

sorted_pairs = sorted(zip(mi, feature_names),reverse=True)
x_sorted, y_sorted = zip(*sorted_pairs)

# Creare il grafico
sns.barplot(x=np.array(x_sorted)[1:Feature_selected], y=np.array(y_sorted)[1:Feature_selected])

plt.xlabel('Mutual Information')

plt.ylabel('Features')

plt.show()

#sns.barplot(x=mi, y=feature_names)


np.random.seed(42)


class LogisticRegressionGD:
    
    def sigmoid_function(self, x): 
        if x >= 0:
            z = np.exp(-x)
            
            return 1/(1+z)
        
        else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
            z = np.exp(x)
            
            return z / (1 + z)
        
    def sigmoid(self, x):
        return np.array([self.sigmoid_function(value) for value in x])
        
        
        
    def loss(self, X, y, weights,bias):  
        z = dot(X, weights)+bias     
        N=len(X)   
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))    
        return -np.sum(predict_1 + predict_0)/N
    
    def gradient_loss(self,X,y,weights,bias):
        z = dot(X, weights)+bias  
        yhat=self.sigmoid(z)
        N=len(X)       
        dw=-1/N*dot(X.T,(y-yhat))
        db=-1/N*np.sum((y-yhat))
        return dw, db
        
    def fit(self,X,y,X2,y2,lr, epochs,threshold=0.5):
        loss=randn(epochs)
        loss_val=randn(epochs)
        bias=rand(1)
        N=X.shape[1]
        weights=rand(N)
        cumulative_time=np.zeros(epochs)
        f1=np.zeros(epochs)
        for k in range(epochs):
            start_time = time.time()
            dw, db=self.gradient_loss(X, y, weights, bias)
            bias=bias-lr*db
            weights=weights-lr*dw
            loss2=self.loss(X, y, weights, bias)
            loss[k]=loss2
            loss2_val=self.loss(X2, y2, weights, bias)
            loss_val[k]=loss2_val
            end_time = time.time()
            
            cumulative_time[k]=cumulative_time[k-1] + (end_time-start_time)
            
            z = dot(X2, weights)+bias
            
            y_pred_i=[1 if i > threshold else 0 for i in self.sigmoid(z)]
            
            f1[k]=f1_score(y2,y_pred_i)
            
        self.weights = weights
        self.bias = bias
        self.loss = loss
        self.loss_val=loss_val
        self.cumulative_time=cumulative_time
        self.f1=f1

    def predict(self,X,threshold):
        z = dot(X, self.weights)+self.bias
        
        return [1 if i > threshold else 0 for i in self.sigmoid(z)], self.loss, self.loss_val, self.cumulative_time, self.f1

lr=0.05

epochs=500

logreg = LogisticRegressionGD()
logreg.fit(X_train, y_train,X_test,y_test,  lr, epochs)

y_pred, loss, loss_val, cumulative_time_gd,f1_gd = logreg.predict(X_test, threshold=0.5)

y_pred=np.array(y_pred)


class NesterovLogisticRegression:
    
    
    def sigmoid_function(self, x): 
        if x >= 0:
            z = np.exp(-x)
            
            return 1/(1+z)
        
        else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
            z = np.exp(x)
            
            return z / (1 + z)
        
    def sigmoid(self, x):
        return np.array([self.sigmoid_function(value) for value in x])
        
        
    def loss(self, X, y, weights,bias):  
        z = dot(X, weights)+bias     
        N=len(X)   
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))    
        return -np.sum(predict_1 + predict_0)/N
    
    def gradient_loss(self,X,y,weights,bias):
        z = dot(X, weights)+bias  
        yhat=self.sigmoid(z)
        N=len(X)       
        dw=-1/N*dot(X.T,(y-yhat))
        db=-1/N*np.sum((y-yhat))
        return dw, db
        
    def fit(self,X,y,X2,y2,lr, epochs,threshold=0.5):
        loss=rand(epochs)
        loss_val=rand(epochs)
        bias=rand(1)
        weights=rand(X.shape[1])
        t=1
        tnew=t
        biaslessone=bias
        weigthslessone=weights
        cumulative_time=np.zeros(epochs)
        f1=np.zeros(epochs)
        for k in range(epochs):
            start_time = time.time()
            tnew=(1+np.sqrt(1+4*t**2))/2
            bias2=bias+(t-1)/tnew*(bias-biaslessone)
            weights2=weights+(t-1)/tnew*(weights-weigthslessone)
            
            dw, db=self.gradient_loss(X, y, weights2, bias2)
            
            weigthslessone=weights
            
            biaslessone=bias
            
            bias=bias2-lr*db
            weights=weights2-lr*dw
            
            
            t=tnew
            
            loss2=self.loss(X, y, weights, bias) 
                        
            loss[k]=loss2
            
            loss2_val=self.loss(X2, y2, weights, bias) 
            
            loss_val[k]=loss2_val
            end_time = time.time()
            
            cumulative_time[k]=cumulative_time[k-1] + (end_time-start_time)

            
            z = dot(X2,weights)+bias
            
            y_pred_i=[1 if i > threshold else 0 for i in self.sigmoid(z)]
            
            f1[k]=f1_score(y2,y_pred_i)
            
            
        
        self.weights = weights
        self.bias = bias
        self.loss = loss
        self.loss_val=loss_val
        self.cumulative_time=cumulative_time
        self.f1=f1
    def predict(self,X,threshold):
        z = dot(X, self.weights)+self.bias
        
        return [1 if i > threshold else 0 for i in self.sigmoid(z)], self.loss,self.loss_val, self.cumulative_time,self.f1
    
logreg = NesterovLogisticRegression()
logreg.fit(X_train, y_train,X_test,y_test,  lr, epochs)

y_pred_nesterov, loss_nesterov,loss_val_nesterov,cumulative_time_nesterov,f1_nesterov = logreg.predict(X_test,threshold=0.5)


class SGDLogisticRegression:
    
    
    def sigmoid_function(self, x): 
        if x >= 0:
            z = np.exp(-x)
            
            return 1/(1+z)
        
        else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
            z = np.exp(x)
            
            return z / (1 + z)
        
    def sigmoid(self, x):
        return np.array([self.sigmoid_function(value) for value in x])
        
        
    def loss(self, X, y, weights,bias):  
        z = dot(X, weights)+bias     
        N=len(X)   
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z)) 
        return -np.sum(predict_1 + predict_0)/N
    
    
    
    def gradient_loss(self,X,y,weights,bias):
        z = dot(X, weights)+bias  
        yhat=self.sigmoid(z)
        N=len(X)       
        dw=-1/N*dot(X.T,(y-yhat))
        db=-1/N*np.sum((y-yhat))
        return dw, db
        
    def fit(self,X,y,X2,y2,lr, epochs,batch_size,threshold=0.5):
        loss=randn(epochs)
        loss_val=randn(epochs)
        bias=rand(1)
        N=X.shape[1]
        weights=rand(N)
        cumulative_time=np.zeros(epochs)
        f1=np.zeros(epochs)
        for k in range(epochs):
            start_time=time.time()
            M=np.random.choice(X.shape[0], size=batch_size)
            
            Xs = X[M,:]
            ys = y[M]
            
            dw, db=self.gradient_loss(Xs, ys, weights, bias)
            bias=bias-lr*db
            weights=weights-lr*dw
            loss2=self.loss(X, y, weights, bias)
            loss[k]=loss2
            loss2_val=self.loss(X2, y2, weights, bias)
            loss_val[k]=loss2_val
            end_time=time.time()
            
            cumulative_time[k]=cumulative_time[k-1]+(end_time-start_time)
            
            z = dot(X2, weights)+bias
            
            y_pred_i=[1 if i > threshold else 0 for i in self.sigmoid(z)]
            
            f1[k]=f1_score(y2,y_pred_i)
            
        self.weights = weights
        self.bias = bias
        self.loss = loss
        self.loss_val = loss_val
        self.cumulative_time=cumulative_time
        self.f1=f1
    def predict(self,X,threshold=0.5):
        z = dot(X, self.weights)+self.bias
        return [1 if i > threshold else 0 for i in self.sigmoid(z)], self.loss, self.loss_val,self.cumulative_time,self.f1


logreg = SGDLogisticRegression()

logreg.fit(X_train, y_train,X_test,y_test,lr,epochs,batch_size=1)

y_SGD, loss_SGD,loss_val_sgd,cumulative_time_sgd,f1_sgd = logreg.predict(X_test,threshold=0.5)

class AdamLogisticRegression:
    
    
    def sigmoid_function(self, x): 
        if x >= 0:
            z = np.exp(-x)
            
            return 1/(1+z)
        
        else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
            z = np.exp(x)
            
            return z / (1 + z)
        
    def sigmoid(self, x):
        return np.array([self.sigmoid_function(value) for value in x])
        
        
    def loss(self, X, y, weights,bias):  
        z = dot(X, weights)+bias     
        N=len(X)   
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))    
        return -np.sum(predict_1 + predict_0)/N
    
    def gradient_loss(self,X,y,weights,bias):
        z = dot(X, weights)+bias  
        yhat=self.sigmoid(z)
        N=len(X)       
        dw=-1/N*dot(X.T,(y-yhat))
        db=-1/N*np.sum((y-yhat))
        return dw, db
        
    def fit(self,X,y,X2,y2,lr, epochs,beta1,beta2,eps,threshold=0.5):
        loss=randn(epochs)
        loss_val=randn(epochs)
        bias=rand(1)
        N=X.shape[1]
        weights=rand(N)
        
        m_w=0
        v_w=0
        
        m_b=0
        v_b=0
        
        t=0
        
        cumulative_time=np.zeros(epochs)
        
        f1=np.zeros(epochs)
        for k in range(epochs):
            
            start_time=time.time()
            t=t+1

            dw, db=self.gradient_loss(X, y, weights, bias)
            
            m_w=beta1*m_w+(1-beta1)*dw
            
            v_w=beta2*v_w+(1-beta2)*np.power(dw,2)
            
            mhat_w=m_w/(1-beta1**t)
            
            vhat_w=v_w/(1-beta2**t)
            
            weights=weights-lr*mhat_w/(np.sqrt(vhat_w)+eps)
            
            
            m_b=m_b*beta1+(1-beta1)*db
            
            v_b=beta2*v_b+(1-beta2)*db**2
            
            mhat_b=m_b/(1-beta1**t)
            
            vhat_b=v_b/(1-beta2**t)
            
            bias=bias-lr*mhat_b/(np.sqrt(vhat_b)+eps)
            
            
            loss2=self.loss(X, y, weights, bias)
            loss[k]=loss2
            
            loss2_validation=self.loss(X2, y2, weights, bias)
            
            loss_val[k]=loss2_validation
            
            end_time=time.time()
            
            cumulative_time[k]=cumulative_time[k-1]+(end_time-start_time)
            
            
            z = dot(X2, weights)+bias
            
            y_pred_i=[1 if i > threshold else 0 for i in self.sigmoid(z)]
            
            f1[k]=f1_score(y2,y_pred_i)
            
            
        self.weights = weights
        self.bias = bias
        self.loss = loss
        self.loss_val=loss_val
        self.cumulative_time=cumulative_time
        self.f1=f1
    def predict(self,X,threshold=0.5):
        z = dot(X, self.weights)+self.bias
        return [1 if i > threshold else 0 for i in self.sigmoid(z)], self.loss,self.loss_val,self.cumulative_time,self.f1

  
  
logreg = AdamLogisticRegression()

logreg.fit(X_train, y_train,X_test,y_test,lr, epochs,beta1=0.9,beta2=0.99,eps=10**(-8))

y_adam, loss_adam,loss_val_adam,cumulative_time_adam,f1_adam = logreg.predict(X_test,threshold=0.5)

fig, axs = plt.subplots(2,figsize=(10, 10))

plt.subplot(2, 1, 1)

sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=loss, label='GD')
sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=loss_nesterov, label='NGD')
sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=loss_SGD, label='SGD')
sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=loss_adam, label='ADAM')

plt.title("Logistic Regression Train Set")

plt.subplot(2, 1, 2)

sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=loss_val, label='GD')
sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=loss_val_nesterov, label='NGD')
sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=loss_val_sgd, label='SGD')
sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=loss_val_adam, label='ADAM')

plt.title("Logistic Regression Validation Set")


fig.show()

logreg=LogisticRegression(penalty=None, solver='sag')

logreg.fit(X_train,y_train)

y_pred_sk=logreg.predict(X_test)

conf_matrix_adam = confusion_matrix(y_test, y_adam)

conf_matrix_gd = confusion_matrix(y_test, y_pred)

conf_matrix_sgd = confusion_matrix(y_test, y_SGD)

conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sk)

conf_matrix_nesterov = confusion_matrix(y_test, y_pred_nesterov)

fig, axs = plt.subplots(1,figsize=(10, 5))

sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=cumulative_time_gd, label='GD')
sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=cumulative_time_nesterov, label='NGD')
sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=cumulative_time_sgd, label='SGD')
sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=cumulative_time_adam, label='ADAM')

plt.xlabel('Iteration')

plt.ylabel('Time')


plt.title("Logistic Regression Training CPU Time along iteration")

fig.show()


fig, axs = plt.subplots(1,figsize=(10, 5))

sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=f1_gd, label='GD')
sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=f1_nesterov, label='NGD')
sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=f1_sgd, label='SGD')
sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=f1_adam, label='ADAM')

plt.xlabel('Iteration')

plt.ylabel('F1-score')


plt.title("F1-score along iteration")

fig.show()

'''
confusion_matrices=([conf_matrix_gd,conf_matrix_sgd,
                              conf_matrix_nesterov,conf_matrix_adam,
                              conf_matrix_sklearn])
'''
plt.figure(5,figsize=(12, 8))



plot_confusion_matrix(conf_matrix_gd, show_absolute=True, show_normed=False)

plt.title('GD')


plot_confusion_matrix(conf_matrix_nesterov, show_absolute=True, show_normed=False)

plt.title('NGD')


plot_confusion_matrix(conf_matrix_sgd, show_absolute=True, show_normed=False)

plt.title('SGD')



plot_confusion_matrix(conf_matrix_adam, show_absolute=True, show_normed=False)

plt.title('ADAM')



plot_confusion_matrix(conf_matrix_sklearn, show_absolute=True, show_normed=False)

plt.title('SKLEARN')

plt.tight_layout()

