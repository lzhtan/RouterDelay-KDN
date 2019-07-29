
# coding: utf-8

# In[1]:


import numpy as np
import re

list_a = []
i = 0
with open('/data/code/RouterDelay/xgb_train_process.txt') as file_object:
    for line in file_object:
        i += 1
        if(i % 2 == 0): 
            list_a.append(line.split())


# In[2]:


array_a = np.asarray(list_a) #list转化成ndarray，便于行列选取
array_a = array_a[:,[1,2]] #取所有行，第2、3列


# In[3]:


list1 = []
for array_1 in array_a: #每行
    list2 = []
    for array_2 in array_1: #按列
        d = re.search("\d+(\.\d+)?", array_2) #提取array_a中的数字
        f = float(d.group())
        list2.append(f)
    list1.append(list2)
    
array_num =  np.asarray(list1) #转换成ndarray，表示标准差mse
array_num = np.square(array_num) #标准差转换为方差

loss = list(array_num[:,0]) #第一列是train_loss
valid_loss =  list(array_num[:,1]) #第二列是valid_loss
