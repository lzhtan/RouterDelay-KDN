
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics
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


# In[6]:


rnd_clf = RandomForestRegressor(n_estimators=100, criterion='mse',max_leaf_nodes=16, n_jobs=-1, verbose=1)
rnd_clf.fit(X_train, y_train) #训练模型


# In[7]:


predict = rnd_clf.predict(X_test) #预测模型
predict_2 = predict + yi_mean #预测值
y_test_2 = y_test + yi_mean #真实值


# In[8]:


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


# In[9]:

#画图展示一下预测值和真实值
nums = range(0,966)
plt.figure(figsize=(20,5))
plt.plot(nums, smooth_curve(predict_2,0), 'r--', label='predict value')
plt.plot(nums, smooth_curve(y_test_2,0), 'b', label='true value')
plt.title('predict value and true value')
plt.legend()
plt.show()


# In[10]:


param_test1= {'n_estimators':range(1,101,1)}
gsearch1= GridSearchCV(estimator = RandomForestRegressor(criterion='mse',
                                                         max_leaf_nodes=16, n_jobs=-1, verbose=0),
                       param_grid =param_test1, scoring='neg_mean_squared_error',cv=2,verbose=2)  #对树的个数进行超参搜索
gsearch1.fit(X_valid,y_valid)


# In[11]:


print(gsearch1.best_params_)
print(gsearch1.best_score_)
print(gsearch1.cv_results_)


# In[12]:


test_loss = gsearch1.cv_results_['mean_test_score'] # list，代表树的数目在test上的loss（其实不一定）
train_loss = gsearch1.cv_results_['mean_train_score'] # list，代表树的数目在train上的loss（其实不一定）
test_loss = np.asarray(test_loss) #为了方便取反，转成ndarray
train_loss = np.asarray(train_loss) #
test_loss = -test_loss #这个loss是负的，为了画图，把它取反
train_loss = -train_loss #
print(test_loss)

