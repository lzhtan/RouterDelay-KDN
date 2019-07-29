
# coding: utf-8

# In[1]:


import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras.layers import Embedding, SimpleRNN
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd


# In[2]:


filename = "/input/KDN_dataset.txt"
data = np.loadtxt(filename)


# In[3]:


n = data.shape[0] #n代表data行数
routing = data[0:n,0:24] #n行，0-23列
traffic = data[0:n,24:168] #n行，24——167列
delay = data[0:n,168:456] #n行，168-455列


# In[4]:


delay_average = delay.mean(1) #288列转为1列，行平均


# In[5]:


Xo = np.concatenate((routing,traffic), axis=1) #列方向连接，n行，0-167列，共24+144=168
#Xo[:,[24,37,50,63,76,89,102,115,128,141,154,167]]元素都为-2.5
Xo = Xo = Xo[:,Xo.std(0) != 0.] #去掉std==0的列
Xo = (Xo - Xo.mean(0))/Xo.std(0)
numFeat = Xo.shape[1] #输入特征数168-12=156
yi = delay_average #yi只有1列
yi_mean = yi.mean()
print(yi_mean)
yi = yi - yi_mean


# In[6]:


# Generate Dataset
"""
划分成3个样本集，训练集Training，验证集Validation，测试集Test
"""
validPercentage = 0.1 #验证集占比
testPercentage = 0.1 #测试集占比

lastTraining = int(n*(1-validPercentage-testPercentage))
lastValid = int(n*(1-testPercentage))

X_train = Xo[0:lastTraining,:] #0-lastTraining行，所有列
y_train = yi[0:lastTraining] #0-lastTraining行，所有列

X_valid = Xo[lastTraining:lastValid,:] #lastTraining-lastValid行，所有列
y_valid = yi[lastTraining:lastValid] #lastTraining-lastValid行，所有列

X_test = Xo[lastValid:n,:] #lastValid-n行，所有列
y_test = yi[lastValid:n] #lastValid-n行，所有列


# In[7]:


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(168, activation='tanh', input_shape=(156,)))
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',
                 loss='mse',
                 metrics=['mae'])
    return model
model = build_model()


# In[8]:


history = model.fit(X_train, y_train, epochs=1000, batch_size=200,validation_data=(X_valid, y_valid)) #返回训练结果


# In[9]:


##数据曲线平滑化##
def smooth_curve(points, factor=0.8): #利用指数加权平均数来平滑曲线
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# In[10]:


#画图展示一下训练过程
import matplotlib.pyplot as plt
mae = history.history['mean_absolute_error'] 
val_mae = history.history['val_mean_absolute_error'] 
loss = history.history['loss'] 
val_loss = history.history['val_loss']

epochs = range(1, len(mae) + 1)

plt.plot(epochs, smooth_curve(mae,0.5), 'r--', label='Training mae') 
plt.plot(epochs, smooth_curve(val_mae,0.5), 'b', label='Validation mae') 
plt.title('Training and validation mean_absolute_error')
plt.legend()

plt.figure()
plt.plot(epochs, smooth_curve(loss,0.5), 'r--', label='Training loss') 
plt.plot(epochs, smooth_curve(val_loss,0.5), 'b', label='Validation loss') 
plt.title('Training and validation loss') 
plt.legend()

plt.show()


# In[11]:


result = model.predict(X_test)
result = result[:,0] #取所有行，第一列

result_2 = result + yi_mean # predict value
y_test_2 = y_test + yi_mean # ture value

#画图展示一下预测值和真实值
nums = range(0,966)
plt.plot(nums, smooth_curve(result_2,0.5), 'r--', label='predict value') 
plt.plot(nums, smooth_curve(y_test_2,0.5), 'b', label='true value') 
plt.title('predict value and true value')
plt.legend()
plt.show()

