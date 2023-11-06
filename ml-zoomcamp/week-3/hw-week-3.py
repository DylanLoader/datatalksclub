#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
# Load data 
car_df = pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv")
# %%
car_df.head()
# %%
retained_cols = [
    "Make",
    "Model",
    "Year",
    "Engine HP", 
    "Engine Cylinders",
    "Transmission Type",
    "Vehicle Style",
    "highway MPG",
    "city mpg",
    "MSRP"
]

car_df = car_df[retained_cols]

# %% Preprocessing 
car_df.columns = car_df.columns.str.replace(' ', '_').str.lower()
car_df.rename(columns={'msrp':'price'}, inplace=True)
car_df.head(10)

# %% Question 1: What is the most frequent observation (mode) for the column transmission_type?
most_common_transmission = car_df['transmission_type'].mode()
print(f"The most common transmission type is {most_common_transmission}")

# %% Question 2: Create the correlation matrix for the numerical features of your dataset. In a correlation matrix, you compute the correlation coefficient between every pair of features in the dataset.

#%%
car_df.head(10)
#%%
# Get the numeric columns 
car_df.dtypes

#%%
# Get all the numeric type columns
num_cols = car_df.select_dtypes(include=np.number).columns.to_list()

# %%
from itertools import combinations

# %%
