#%%
# ML Zoomcamp Week 2 homework 2023
#%% Pull data and imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv")

#%%
df.head()

#%% Preliminary Filtering

# First, keep only the records where ocean_proximity is either '<1H OCEAN' or 'INLAND'
df = df[df.ocean_proximity.isin(['<1H OCEAN', 'INLAND'])]

# Keep only the following columns: 'latitude', 'longitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value'.
retained_cols = [
    'latitude', 
    'longitude', 
    'housing_median_age', 
    'total_rooms', 
    'total_bedrooms', 
    'population', 
    'households', 
    'median_income', 
    'median_house_value'
                 ]

df = df[retained_cols]
#%% Question 1 There's one feature with missing values. What is it?
df.isna().sum()

#%% Question 1 Answer
print(f"Only the feature missing values is total_bedrooms, it is missing {df['total_bedrooms'].isna().sum()} values.")

#%% Question 2 What's the median (50% percentile) for variable 'population'?
population_median = df['population'].median()

print(f"Question 2 Answer: The median population is: {population_median}")

#%% Question 3

#Shuffle the dataset (the filtered one you created above), use seed 42.
seed = 42
df_shuffled = df.sample(
    frac=1, 
    random_state=seed
    )
# Split your data in train/val/test sets, with 60%/20%/20% distribution.
df_train = df_shuffled.iloc[:int(df_shuffled.shape[0]*.6)].reset_index(drop=True)
df_val = df_shuffled.iloc[int(df_shuffled.shape[0]*.6):int(df_shuffled.shape[0]*.8)].reset_index(drop=True)
df_test = df_shuffled.iloc[int(df_shuffled.shape[0]*.8):].reset_index(drop=True)

assert df_shuffled.shape[0] == (df_train.shape[0] + df_val.shape[0] + df_test.shape[0])

# Apply the log transformation to the median_house_value variable using the np.log1p() function.
# Since log transform is monotonic and constant across all datasets we can do this transform in place. 
y_train = np.log1p(df_train['median_house_value'])
y_val = np.log1p(df_val['median_house_value'])
y_test = np.log1p(df_test['median_house_value'])

# remove the targets from their respective datasets
del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']


# Imputation 
def impute_data(df_train:pd.DataFrame, df_val:pd.DataFrame, df_test:pd.DataFrame, col:str='total_bedrooms', method:str='mean')-> pd.DataFrame:
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()
    if method == 'mean':
        train_mean = df_train[col].mean()
        df_train[col].fillna(value=train_mean, inplace=True)
        df_val[col].fillna(value=train_mean, inplace=True)
        df_test[col].fillna(value=train_mean, inplace=True)

    elif method == 'zero':
        df_train[col].fillna(value=0, inplace=True)
        df_val[col].fillna(value=0, inplace=True)
        df_test[col].fillna(value=0, inplace=True)
    return df_train, df_val, df_test
    
df_train_mean_imputed, df_val_mean_imputed, df_test_mean_imputed = impute_data(df_train, df_val, df_test, method='mean')
df_train_zero_imputed, df_val_zero_imputed, df_test_zero_imputed = impute_data(df_train, df_val, df_test, method='zero')

assert df_train_mean_imputed.total_bedrooms.sum() != df_train_zero_imputed.total_bedrooms.sum()

#%% 
df_train_mean_imputed.total_bedrooms.sum()

#%%
df_train_zero_imputed.total_bedrooms.sum()
#%%
df_train_zero_imputed.total_bedrooms.isna().sum()
# %%
df_train_mean_imputed
# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr_mean = LinearRegression()
lr_mean.fit(df_train_mean_imputed, y_train)
y_val_pred = lr_mean.predict(df_val_mean_imputed)

#%% 
rmse_mean_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
rmse_mean_val

#%% 
###
lr_zero = LinearRegression()
lr_zero.fit(df_train_zero_imputed, y_train)
y_val_pred = lr_mean.predict(df_val_zero_imputed)

#%% 
rmse_zero_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
rmse_zero_val

print(f"The rmse for zero imputation is {rmse_zero_val:.02f} while the rmse for mean imputation is {rmse_mean_val:.02f}")
print("The difference between the two imputation methods is negligible")
# %% Question 4
# Using the method of regularization from the youtube video 2.13
# 
def train_linear_regression_reg(X, y, alpha=0.1):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX = XTX + alpha*np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)   
    return w_full[0], w_full[1:]

X_train = df_train_zero_imputed.values
X_val = df_val_zero_imputed.values
min_rmse = 1000
alpha_arr = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
for alpha in alpha_arr:
    w0, w = train_linear_regression_reg(X_train, y_train,alpha=alpha)
    y_pred = w0 + X_val.dot(w)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred))
    min_rmse = min(rmse_val, min_rmse)
    print(f"For the regularization parameter {alpha} the corresponding RMSE is: {rmse_val}")
print(f"The minimum RMSE is {min_rmse}")

print(f"The smallest RMSE occurs when we apply no regularization in this case. ")
# %% Question 5
seed_arr = [0,1,2,3,4,5,6,7,8,9]
rmse_dict = {}
for idx, seed in enumerate(seed_arr):
    #Shuffle the dataset (the filtered one you created above), use seed 42.
    df_shuffled = df.sample(
        frac=1, 
        random_state=seed
        )
    # Split your data in train/val/test sets, with 60%/20%/20% distribution.
    df_train = df_shuffled.iloc[:int(df_shuffled.shape[0]*.6)].reset_index(drop=True)
    df_val = df_shuffled.iloc[int(df_shuffled.shape[0]*.6):int(df_shuffled.shape[0]*.8)].reset_index(drop=True)
    df_test = df_shuffled.iloc[int(df_shuffled.shape[0]*.8):].reset_index(drop=True)
    
    # Apply log transform
    y_train = np.log1p(df_train['median_house_value'])
    y_val = np.log1p(df_val['median_house_value'])
    y_test = np.log1p(df_test['median_house_value'])

    # remove the targets from their respective datasets
    del df_train['median_house_value']
    del df_val['median_house_value']
    del df_test['median_house_value']
    
    df_train_zero_imputed, df_val_zero_imputed, df_test_zero_imputed = impute_data(df_train, df_val, df_test, method='zero')
    # Create linear_model
    lr = LinearRegression()
    lr.fit(df_train_zero_imputed, y_train)
    y_pred = lr.predict(df_val_zero_imputed)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_dict[f"seed_{idx}"] = rmse_val
# %%
rmse_dict
# %%
seed_std = np.std(list(rmse_dict.values()))
# %%
print(f"The standard deviation of the RMSE is {seed_std:.04f}")
# %% Question 6

# Split the dataset like previously, use seed 9.
seed = 9 
#Shuffle the dataset (the filtered one you created above), use seed 42.
df_shuffled = df.sample(
    frac=1, 
    random_state=seed
    )
# Split your data in train/val/test sets, with 60%/20%/20% distribution.
df_train = df_shuffled.iloc[:int(df_shuffled.shape[0]*.8)].reset_index(drop=True)
df_test = df_shuffled.iloc[int(df_shuffled.shape[0]*.8):].reset_index(drop=True)

#%%
df_train.shape
#%%
df_test.shape
#%%
# Apply log transform
y_train = np.log1p(df_train['median_house_value'])
y_test = np.log1p(df_test['median_house_value'])

y_train.shape[0]+y_test.shape[0]
#%%
# remove the targets from their respective datasets
del df_train['median_house_value']
del df_test['median_house_value']
#%% 
# Fill the missing values with 0 and train a model with r=0.001
def impute_data(df_train:pd.DataFrame, df_test:pd.DataFrame, col:str='total_bedrooms', method:str='mean')-> pd.DataFrame:
    df_train = df_train.copy()
    df_test = df_test.copy()
    if method == 'mean':
        train_mean = df_train[col].mean()
        df_train[col].fillna(value=train_mean, inplace=True)
        df_test[col].fillna(value=train_mean, inplace=True)

    elif method == 'zero':
        df_train[col].fillna(value=0, inplace=True)
        df_test[col].fillna(value=0, inplace=True)
    return df_train, df_test

df_train_comb, df_test_comb = impute_data(df_train, df_test, method='zero')

#%%
df_train_comb.shape, df_test_comb.shape
#%%
assert df.shape[0] == df_train_comb.shape[0]+df_test_comb.shape[0]

# %%
def train_linear_regression_reg(X, y, alpha=0.1):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX = XTX + alpha*np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)   
    return w_full[0], w_full[1:]

#%%
y_train.shape
#%%%
w0, w = train_linear_regression_reg(df_train_comb.values, y_train, alpha=0.001)
y_pred = w0 + df_test_comb.values.dot(w)
#%%
y_pred
#%%
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_test

# %%
# What's the RMSE on the test dataset?