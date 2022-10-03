#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns


# ## Read dataset

# In[2]:


# Read data used for training and test

train = pd.read_csv('../Dataset/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../Dataset/house-prices-advanced-regression-techniques/test.csv')


# ## Exploratory data analysis

# In[3]:


train.head()


# A potential issue here is that "NA" in "Alley", and in other categories is interpreted as "NaN", hence missing data, although is isn't

# In[4]:


test.head()


# Id it's useless, I'm going to drop it!

# In[5]:


train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)


# Let's look at the correlations of the data

# In[6]:


correlations_train = train.corr()


# In[7]:


sns.set(font_scale=1.4)


# Which variables have stronger correlation with the Sale price that we want to predict?

# In[8]:


plt.figure(figsize=(20, 10))
heatmap = sns.heatmap(correlations_train[['SalePrice']].sort_values(by='SalePrice', ascending=False)[:10], vmin=-1, vmax=1, annot=True, cmap='magma',cbar=False)
heatmap.set_title('Correlation of features with Sales Price', fontdict={'fontsize':18}, pad=16);


# This exploratory analysis revealed the most important features that drive the house prize. They are:
# - Quality of the house, 
# - Size,
# - Cars in garage
# - Bathrooms,
# - etc...
# 
# Some of these feature are likely strongly correlated, such as "GarageCars" and "GarageArea"

# Let's explore more this correlation between features

# In[50]:


mask = np.triu(correlations_train)
plt.figure(figsize=(20, 20))
sns.heatmap(correlations_train,
            annot=True,
            fmt='.1f',
            cmap='magma',
            square=True,
            mask=mask,
            linewidths=1,
            cbar=False)

plt.show()


# This confirms that variables like "GarageArea" and "GarageCars" are strongly correlated. Maybe it makes sense to consider using only one of them and reduce the dimensionality of the problem

# ## Data preparation

# Split dataset in X and y

# In[20]:


y = train['SalePrice'].reset_index(drop=True)
X = train.drop(['SalePrice'], axis=1)


# Let's deal with missing data first

# In[23]:


def fraction_missing_data(df):
    
    """Compute the fraction of missing data for each variable."""
    
    # Compute the number of missing data per variable
    total = df.isnull().sum().sort_values(ascending=False)
    
    # Mask to filter cases with no missing data
    mask_missing_data = total != 0
    
    # Compute fraction of missing data
    percent = (total / len(df) * 100)[mask_missing_data]
    
    # Apply mask to filter cases with no missing data
    total = total[mask_missing_data]
    percent = percent[mask_missing_data]
    
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# In[24]:


missing_data = fraction_missing_data(X)


# In[38]:


fig, ax = plt.subplots(figsize=(20, 5))
sns.barplot(x=missing_data.index, y='Percent', data=missing_data, palette='Oranges_r')
plt.xticks(rotation=90)
plt.show()


# In[42]:


X[missing_data.index]


# As anticipated, for some categories we seem to have missing data. In reality pandas is just misinterpreting the "NA"s.
# Let's take care of them first

# In[44]:


# Categories where 'NaN' mean None.

none_cols = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',
            'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
]

for col in none_cols:
    X[col].replace(np.nan, 'None', inplace=True)


# In[46]:


missing_data = fraction_missing_data(X)
fig, ax = plt.subplots(figsize=(20, 5))
sns.barplot(x=missing_data.index, y='Percent', data=missing_data, palette='Oranges_r')
plt.xticks(rotation=90)
plt.show()


# In[47]:


X[missing_data.index]


# In[3]:


# Find the categorical columns

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(train)


# Split variables in dependent and independent

X = train.iloc[:,:-1].values
y = train.iloc[:,-1].values

# OneHot encoding of the categorical variables
## Find indices associated to categorical variables
indices_of_categorical_columns = np.array([np.where(train.keys() == key)[0][0] for key in categorical_columns])
## Apply OneHot encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), indices_of_categorical_columns)], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[4]:


# Split variables in training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[5]:


# Imputing missing values
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)


# ## Fit using random forest regression

# In[6]:


from sklearn.ensemble import RandomForestRegressor


# In[7]:


rfr = RandomForestRegressor(n_estimators=10,random_state=0)


# In[8]:


rfr.fit(X_train,y_train)


# ## Compare against the test set

# In[9]:


from sklearn.metrics import r2_score


# In[10]:


y_predict = rfr.predict(X_test)
r2 = r2_score(y_test,y_predict)


# In[11]:


plt.figure(figsize=(10,10))
plt.scatter(y_test/10**3,y_predict/10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('True house prize [k$]')
plt.ylabel('Predicted house prize [k$]')
plt.text(70,600,f'R^2 = {r2}')
plt.xlim(60,800)
plt.ylim(60,800)


# In[12]:


fractional_difference = np.abs((y_predict-y_test)/y_test)
plt.figure(figsize=(10,10))
bins = np.logspace(np.log10(np.min(fractional_difference)),np.log10(np.max(fractional_difference)),num=int(len(fractional_difference)/30))
plt.hist(fractional_difference,bins=bins,histtype='step')
plt.axvline(np.median(fractional_difference))
plt.xscale('log')
plt.xlabel('fractional difference')
plt.ylabel('counts')


# In[ ]:




