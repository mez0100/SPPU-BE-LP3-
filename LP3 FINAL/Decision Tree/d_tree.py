#!/usr/bin/env python
# coding: utf-8

# In[9]:

#In Anaconda CMD, install: conda install python-graphviz
#In Jupyter Notebook install: conda install pydotplus

import numpy as np
import pandas as pd


# In[10]:


dataset=pd.read_csv("C:/Users/Admin/Downloads/ml (1)/d_tree/income.csv")
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1].values


# In[11]:


#perform label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()


# In[12]:


X=X.apply(LabelEncoder().fit_transform)
print(X)


# In[13]:


from sklearn.tree import DecisionTreeClassifier
regressor=DecisionTreeClassifier() #object
regressor.fit(X.iloc[:,1:5],y)


# In[14]:


#predict value for the given expression
X_in=np.array([1,1,0])

y_pred=regressor.predict([X_in])
print("Prediction :", y_pred)


# In[15]:


import six
import sys
sys.modules['sklearn.externals.six']=six
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()

export_graphviz(regressor, out_file=dot_data, filled=True, rounded=True,
               special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree1.png')


# In[ ]:




