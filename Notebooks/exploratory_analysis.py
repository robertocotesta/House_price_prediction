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

# Let's explore more this correlation between numerical features

# In[9]:


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

# Let's look now at the importance of categorical variables for the estimating price

# In[10]:


from matplotlib.ticker import MaxNLocator
def boxplots_cat(y, df):
    
    '''Create boxplots for categorical variables'''
    
    ncol = 3
    nrow = int(np.ceil(len(df.select_dtypes(include=['object']).columns)/ncol))
    
    fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 5*nrow))
    axes = axes.flatten()

    for i, j in zip(df.select_dtypes(include=['object']).columns, axes):

        sortd = df.groupby([i])[y].median().sort_values(ascending=False)
        sns.boxplot(x=i,
                    y=y,
                    data=df,
                    palette='magma',
                    order=sortd.index,
                    ax=j)
        j.tick_params(labelrotation=45)
        j.yaxis.set_major_locator(MaxNLocator(nbins=18))

        plt.tight_layout()


# In[11]:


boxplots_cat('SalePrice', train)


# Some categories like "LotConfig" and "LandSlope" don't seem to make a big difference. Maybe it's worth thinking about
# reducing the dimensionality of the model by removing them

# ## Data preparation

# Split dataset in X and y

# In[12]:


y = train['SalePrice'].reset_index(drop=True)
X = train.drop(['SalePrice'], axis=1)
X = pd.concat([X,test]).reset_index(drop=True)


# Let's deal with missing data first

# In[13]:


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


# In[14]:


missing_data = fraction_missing_data(X)


# In[15]:


fig, ax = plt.subplots(figsize=(20, 5))
sns.barplot(x=missing_data.index, y='Percent', data=missing_data, palette='Oranges_r')
plt.xticks(rotation=90)
plt.show()


# In[16]:


X[missing_data.index]


# As anticipated, for some categories we seem to have missing data. In reality pandas is just misinterpreting the "NA"s.
# Let's take care of them first

# In[17]:


# Categories where 'NaN's mean None.

none_cols = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',
            'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
]

for col in none_cols:
    X[col].replace(np.nan, 'None', inplace=True)


# In[18]:


missing_data = fraction_missing_data(X)
fig, ax = plt.subplots(figsize=(20, 5))
sns.barplot(x=missing_data.index, y='Percent', data=missing_data, palette='Oranges_r')
plt.xticks(rotation=90)
plt.show()


# In[19]:


X[missing_data.index]


# In[20]:


# Categories where 'NaN's mean 0.

zero_cols = [
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
    'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea'
]    

for col in zero_cols:
    X[col].replace(np.nan, 0, inplace=True)


# In[21]:


missing_data = fraction_missing_data(X)
fig, ax = plt.subplots(figsize=(20, 5))
sns.barplot(x=missing_data.index, y='Percent', data=missing_data, palette='Oranges_r')
plt.xticks(rotation=90)
plt.show()


# In[22]:


X[missing_data.index]


# In[23]:


# Categories where 'NaN's are actually missing data. I replace the categorical ones with the mode
# the numerical one (LotFrontage) I replace with the median

freq_cols = [
    'Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual',
    'SaleType', 'Utilities','MSZoning'
]


# In[24]:


for col in freq_cols:
    X[col].replace(np.nan, X[col].mode()[0], inplace=True)

X['LotFrontage'].replace(np.nan, X['LotFrontage'].median(), inplace=True)


# Let's drop the useless categories I found before

# In[25]:


X.drop(['LotConfig','LandSlope','GarageArea'],axis=1,inplace=True)


# Let's find the categorical variables to encode them

# In[26]:


# Find the categorical columns

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(X)

# OneHot encoding of the categorical variables
## Find indices associated to categorical variables
indices_of_categorical_columns = np.array([np.where(X.keys() == key)[0][0] for key in categorical_columns])
## Apply OneHot encoding
X = X.values
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), indices_of_categorical_columns)], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# After taking care of missing data, let's plit again Xs in train and test set for submission

# In[27]:


X = X[:len(y), :]
X_test_submission = X[len(train):, :]


# Convert the house price in its log, as we want to optimize this quantity

# In[28]:


y = np.log10(y.values)


# In[29]:


# Split variables in training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ## Fit using random forest regression

# In[30]:


from sklearn.ensemble import RandomForestRegressor


# In[31]:


rfr = RandomForestRegressor(n_estimators=10,random_state=0)


# In[32]:


rfr.fit(X_train,y_train)


# ## Compare against the test set

# In[33]:


from sklearn.metrics import r2_score


# In[34]:


y_predict = rfr.predict(X_test)
r2 = r2_score(y_test,y_predict)


# In[35]:


plt.figure(figsize=(10,10))
plt.scatter(10**(y_test)/10**3,10**(y_predict)/10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('True house prize [k$]')
plt.ylabel('Predicted house prize [k$]')
plt.text(70,600,f'R^2 = {r2}')
plt.xlim(60,800)
plt.ylim(60,800)


# In[36]:


fractional_difference = np.abs((y_predict-y_test)/y_test)
plt.figure(figsize=(10,10))
bins = np.logspace(np.log10(np.min(fractional_difference)),np.log10(np.max(fractional_difference)),num=int(len(fractional_difference)/30))
plt.hist(fractional_difference,bins=bins,histtype='step')
plt.axvline(np.median(fractional_difference))
plt.xscale('log')
plt.xlabel('fractional difference')
plt.ylabel('counts')


# In[ ]:




