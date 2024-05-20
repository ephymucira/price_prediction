#!/usr/bin/env python
# coding: utf-8

# In[356]:


import pandas as pd
import matplotlib.pyplot as plt


# In[357]:


df = pd.read_csv('C:/Users/HP 840 G3/Downloads/archive (3)/Copy of survey_results_public.csv')


# In[358]:


df.head()


# In[359]:


df = df[["Country","EdLevel","YearsCodePro","Employment","ConvertedComp"]]
df = df.rename({"ConvertedComp":"Salary"}, axis=1)
df.head()


# In[360]:


df = df[df["Salary"].notnull()]
df.head()


# In[361]:


df.info()


# In[362]:


df = df.dropna()
df.isna().sum()


# In[363]:


df = df[df["Employment"]== "Employed full-time"]
df = df.drop("Employment", axis = 1)
df.info()


# In[364]:


df


# In[365]:


df['Country'].value_counts()


# getting rid of the countries with 1

# In[366]:


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]]= categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'other'
    return categorical_map


# In[367]:


country_map = shorten_categories(df.Country.value_counts(), 400)
df['Country'] = df['Country'].map(country_map)
df.Country.value_counts()


# In[368]:


fig, ax = plt.subplots(1,1, figsize=(12,7))
df.boxplot('Salary','Country', ax=ax)
plt.subtitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


# In[369]:


df= df[df['Salary'] <= 250000]
df= df[df['Salary'] >= 10000]
df= df[df['Country'] != 'other']


# In[370]:


fig, ax = plt.subplots(1,1, figsize=(12,7))
df.boxplot('Salary','Country', ax=ax)
plt.subtitle('Salary (US$) v Country')
plt.title('')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()


# In[371]:


df['YearsCodePro'].unique()


# In[372]:


def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)


# In[373]:


df['EdLevel']


# In[374]:


df['EdLevel'].unique()


# In[375]:


def clean_education(x):
    # Convert to lowercase
    x = x.replace("Master’s degree (MA, MS, M.Eng., MBA, etc.)", "Masters degree")
    x = x.replace("Bachelor’s degree (BA, BS, B.Eng., etc.)", "Bachelors degree")
    x= str(x)
    
    if "Bachelors degree" in x:
        return "Bachelors degree"
    elif "Masters degree" in x:
        return "Masters degree"
    elif "Professional degree (JD, MD, etc.)" in x or 'Other doctoral degree (Ph.D, Ed.D., etc.)' in x:
        return "Post grad"
    else:
        return "Less than a Bachelors"
        
    # If none of the conditions match, return the original value
    

df['EdLevel'] = df['EdLevel'].apply(clean_education)


# In[376]:


df.head()


# In[377]:


df.tail()


# In[378]:


df['EdLevel'].unique()
#the function above aint working as expexted


# In[379]:


from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
df['EdLevel'].unique()


# In[380]:


df.head()


# In[381]:


le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
df['Country'].unique()


# In[382]:


X = df.drop("Salary", axis =1)
y = df["Salary"]


# In[383]:


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X, y.values)
y_pred = linear_reg.predict(X)


# In[384]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
error = np.sqrt(mean_squared_error(y, y_pred))


# In[385]:


error


# In[386]:


from sklearn.tree import DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(X, y.values)


# In[387]:


y_pred = dec_tree_reg.predict(X) 


# In[388]:


error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))
#error


# In[389]:


from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor(random_state= 0)
random_forest_reg.fit(X, y.values)


# In[390]:


y_pred = random_forest_reg.predict(X) 


# In[391]:


error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# In[392]:


from sklearn.model_selection import GridSearchCV

max_depth = [None, 2,4,6,8,10,12]
parameters = {"max_depth": max_depth}

regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y.values)


# In[393]:


regressor = gs.best_estimator_

regressor.fit(X, y.values)
y_pred = regressor.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))


# In[394]:


X


# In[395]:


X = np.array([["United States", 'Masters degree',15]])
X


# In[396]:


X[:, 0] = le_country.transform(X[:,0])
X[:, 1] = le_education.transform(X[:,1])
X = X.astype(float)
X


# In[397]:


y_pred = regressor.predict(X)
y_pred


# In[398]:


import pickle


# In[405]:


data = {"model": regressor, "le_country": le_country, "le_education":le_education}
try:
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
        regressor_loaded = data['model']
        le_country = data["le_country"]
        le_education = data["le_education"]
        
# Handle the case where the file is empty or doesn't exist
except (EOFError, FileNotFoundError):
    # If the file is empty or doesn't exist, create the necessary data
        regressor_loaded = data['model']
        le_country = data["le_country"]
        le_education = data["le_education"]  # Placeholder for education label encoder


# In[406]:


y_pred = regressor_loaded.predict(X)
y_pred


# In[ ]:




