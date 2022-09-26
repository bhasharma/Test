#!/usr/bin/env python
# coding: utf-8

# # Load Wisconsin Breast Cancer Dataset

# In[11]:


import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


data = load_breast_cancer()
X = data.data
y = data.target
print(X.shape, data.feature_names)


# # Classification Model Evaluation Metrics

# In[12]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape)


# In[13]:


from sklearn import linear_model
#max_iter=2600
# solver='newton-cg' or 'liblinear'
logistic = linear_model.LogisticRegression(solver='newton-cg')
logistic.fit(X_train,y_train)


# ## Confusion Matrix

# In[14]:


import model_evaluation_utils as meu
import pandas as pd

y_pred = logistic.predict(X_test)
meu.display_confusion_matrix(true_labels=y_test, predicted_labels=y_pred, classes=[0, 1])


# ## True Positive, False Positive, True Negative and False Negative

# In[15]:


positive_class = 1
TP = 106
FP = 4
TN = 59
FN = 2


# ## Accuracy

# In[16]:


fw_acc = round(meu.metrics.accuracy_score(y_true=y_test, y_pred=y_pred), 5)
mc_acc = round((TP + TN) / (TP + TN + FP + FN), 5)
print('Framework Accuracy:', fw_acc)
print('Manually Computed Accuracy:', mc_acc)


# ## Precision

# In[17]:


fw_prec = round(meu.metrics.precision_score(y_true=y_test, y_pred=y_pred), 5)
mc_prec = round((TP) / (TP + FP), 5)
print('Framework Precision:', fw_prec)
print('Manually Computed Precision:', mc_prec)


# ## Recall

# In[18]:


fw_rec = round(meu.metrics.recall_score(y_true=y_test, y_pred=y_pred), 5)
mc_rec = round((TP) / (TP + FN), 5)
print('Framework Recall:', fw_rec)
print('Manually Computed Recall:', mc_rec)


# ## F1-Score

# In[19]:


fw_f1 = round(meu.metrics.f1_score(y_true=y_test, y_pred=y_pred), 5)
mc_f1 = round((2*mc_prec*mc_rec) / (mc_prec+mc_rec), 5)
print('Framework F1-Score:', fw_f1)
print('Manually Computed F1-Score:', mc_f1)


# ## ROC Curve and AUC

# In[20]:


meu.plot_model_roc_curve(clf=logistic, features=X_test, true_labels=y_test)


# In[ ]:





# In[ ]:




