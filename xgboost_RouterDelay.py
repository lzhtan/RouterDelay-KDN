
# coding: utf-8

# In[1]:


import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


filename = "/input/KDN_dataset.txt"
data = np.loadtxt(filename)


# In[3]:


n = data.shape[0] #n代表data行数
routing = data[0:n,0:24] #n行，0-23列
traffic = data[0:n,24:168] #n行，24——167列
delay = data[0:n,168:456] #n行，168-455列
delay_average = delay.mean(1) #288列转为1列，行平均


# In[4]:


Xo = np.concatenate((routing,traffic), axis=1) #列方向连接，n行，0-167列，共24+144=168
#Xo[:,[24,37,50,63,76,89,102,115,128,141,154,167]]元素都为-2.5
Xo = Xo = Xo[:,Xo.std(0) != 0.] #去掉输入特征里面std==0的列
Xo = (Xo - Xo.mean(0))/Xo.std(0) #对输入特征归一化
numFeat = Xo.shape[1] #输入特征数168-12=156
yi = delay_average #yi只有1列
yi_mean = yi.mean() #yi均值为0
print(yi_mean)
yi = yi - yi_mean


# In[5]:


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


# In[9]:


data_train = xgb.DMatrix(X_train, y_train)
data_valid = xgb.DMatrix(X_valid, y_valid)
data_test = xgb.DMatrix(X_test, y_test)


# In[10]:


param = {}
# use softmax multi-class classification
param['objective'] = 'reg:linear'
# scale weight of positive examples
param['eval_metic']= 'mse'

#param['silent'] = 1
param['verbosity'] = 0 #详细程度=info
param['eta'] = 0.1
param['max_depth'] = 6
param['subsample'] = 0.9

watchlist = [(data_train, 'loss'), (data_valid, 'loss_valid')]


# In[11]:


bst = xgb.train(param, data_train, 1000, watchlist) #xgb_train_process.txt来源于xgb训练过程中的输出


# In[12]:


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


# In[13]:


predict = bst.predict(data_test)
true = data_test.get_label()
predict_2 = predict + yi_mean #真实值
true_2 = true + yi_mean #预测值


# In[14]:


#画图展示一下预测值和真实值
nums = range(0,966)
plt.plot(nums, smooth_curve(predict_2,0.5), 'r--', label='predict value') 
plt.plot(nums, smooth_curve(true_2,0.5), 'b', label='true value') 
plt.title('Training and validation loss') 
plt.legend()
plt.show()