import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
import matplotlib.pyplot as plt
onehot_encoder = OneHotEncoder(sparse_output=False)
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from mlxtend.plotting import plot_confusion_matrix


class MultiClass: 
                  
    def loss(self,X,W,Y,Bias):
        Z=-(X@W+Bias)
        N=len(X)
        l2=np.sum(np.log(np.sum(np.exp(Z),axis=1)))
        l1=np.trace((X @ W+Bias) @ Y.T)
        
        return 1/N*(l1+l2)
                                   
  
    def fit(self,X,Y,X2,Y2,epochs,lr):
        Y_onehot = onehot_encoder.fit_transform(Y.reshape(-1,1))
        if ((X2 is not None) & (Y2 is not None)):
            Y_onehot2 = onehot_encoder.fit_transform(Y2.reshape(-1,1))
            
        losss2=np.full(epochs, np.nan)
        W=np.random.rand(X.shape[1],Y_onehot.shape[1])
        Bias=np.random.rand(1,W.shape[1])
        print(Bias)
        losss=np.random.rand(epochs)
        m_w=0
        v_w=0
        
        m_w2=0

        v_w2=0
        
        
        beta1=0.9
        beta2=0.99
        eps=10**(-8)
        t=0
        for k in range(epochs):
            t=t+1
             
            
            P = softmax(-X@W-Bias, axis=1)
            N = X.shape[0]
            dw = 1/N * (X.T @ (Y_onehot - P))
            
            db = 1/N * np.sum((Y_onehot - P),axis=0)

            m_w=beta1*m_w+(1-beta1)*dw
            
            v_w=beta2*v_w+(1-beta2)*np.power(dw,2)
            
            mhat_w=m_w/(1-beta1**t)
            
            vhat_w=v_w/(1-beta2**t)
            
            W=W-lr*mhat_w/(np.sqrt(vhat_w)+eps)
            
            m_w2=beta1*m_w2+(1-beta1)*db
            
            v_w2=beta2*v_w2+(1-beta2)*np.power(db,2)
            
            mhat_w2=m_w2/(1-beta1**t)
            
            vhat_w2=v_w2/(1-beta2**t)
            
            Bias=Bias-lr*mhat_w2/(np.sqrt(vhat_w2)+eps)
                        
            loss2=self.loss(X,W,Y_onehot,Bias)
            losss[k]=loss2
            
            if ((X2 is not None) & (Y2 is not None)):
            
                loss22=self.loss(X2,W,Y_onehot2,Bias)
                losss2[k]=loss22
                
        self.W=W
        self.losss=losss
        self.losss2=losss2
        self.Bias=Bias
        
    
    def predict(self,X):
        
         P = softmax(-X@self.W-self.Bias, axis=1)
        
         return np.argmax(P,axis=1), self.losss,self.losss2,self.Bias

    
X = load_digits()['data']
y = load_digits()['target']
feature_names = load_digits()['feature_names'] 

scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

mi=mutual_info_classif(X_train, y_train, discrete_features='auto',random_state=42)



sorted_indices = np.argsort(mi)[::-1]

Feature_selected=100

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

model = MultiClass()

epochs=200


model.fit(X_train, y_train,X_test,y_test,epochs,lr=0.05)

# predict 
y_pred,loss,loss_val,b=model.predict(X_test)
        
conf_matrix = confusion_matrix(y_test, y_pred)

sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=loss,label='Train Set')

sns.lineplot(x=np.linspace(0,epochs+1,epochs),y=loss_val, label='Validation Set')

fig,  ax = plt.subplots(figsize=(15, 5))

plot_confusion_matrix(conf_matrix, show_absolute=True, show_normed=False)

