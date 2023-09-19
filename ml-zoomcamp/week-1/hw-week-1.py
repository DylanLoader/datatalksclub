# Homework week 1
# Imports 
#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
df = pd.read_csv("./housing.csv")

# %%
# Question 2:
print(f"There are {df.shape[1]} columns in the dataset")
# %%
# Qustion 3: Which Columns in the dataset have missing values
df.info(verbose=True, show_counts=True)
# %%
# We can also write the query as follows
print(f"There are missing values in the column(s): {df.columns[df.isna().any()].values}.")
# %%
# Question 4: How many unique values does the ocean_proximity column have?
print(f"There are {df.ocean_proximity.nunique()} unique values in the `ocean_proximity` column.")
# %%
# Question 5: What's the average value of the median_house_value for the houses located near the bay?

#%%
df.head()
# %%
print(f"The average value of `median_house_value` for houses near the bay is {np.mean(df[df.ocean_proximity=='NEAR BAY']['median_house_value']):.0f}")
# %%
# Question 6
#    Calculate the average of total_bedrooms column in the dataset.
avg_total_bedroom_unfilled = np.mean(df.total_bedrooms)
#    Use the fillna method to fill the missing values in total_bedrooms with the mean value from the previous step.
df['total_bedrooms_filled'] = df['total_bedrooms'].fillna(value=avg_total_bedroom_unfilled                                                   )
#    Now, calculate the average of total_bedrooms again.
avg_total_bedroom_filled = np.mean(df.total_bedrooms_filled)
#    Has it changed?
print(f"Before filling the values the mean number of total bedrooms is {avg_total_bedroom_unfilled:.3f}, after filling it is {avg_total_bedroom_filled:.3f}")
# To 3 deimal placed, no but this isn[t sureprising considering there are relatively few missing values and we are imputing the mean]

# We can take this a step further and look at the distribution of the total_bedrooms feature

#%% 
total_bedroom_mean = df.total_bedrooms.mean()
total_bedroom_std = df.total_bedrooms.std()

print(f"The mean of total bedrooms for the sample is: {total_bedroom_mean:.02f}, the standard deviatioin is {total_bedroom_std:.02f}")
# %%
df.total_bedrooms.hist(bins=100)
plt.axvline(x=df.total_bedrooms.mean(),
            color='red')
plt.axvline(x=df.total_bedrooms.median(),
            color='blue', ls="--")

plt.show()

# %%
# Question 7
#Select all the options located on islands.
X = df.query('ocean_proximity == "ISLAND"').filter(items= ['housing_median_age', 'total_rooms', 'total_bedrooms']).values
# Select only columns housing_median_age, total_rooms, total_bedrooms.
#Get the underlying NumPy array. Let's call it X.
#Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
#Compute the inverse of XTX.
matrix_prod = X.T@X
#Create an array y with values [950, 1300, 800, 1000, 1300].
y = np.array([950, 1300, 800, 1000, 1300])
#Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
w = np.linalg.inv(matrix_prod)@y
#What's the value of the last element of w?
# w[-1]
#%% 
y

# %%
df.head()
# %%
a = np.ones([9, 5, 7, 4])

c = np.ones([9, 5, 4, 3])
np.dot(a, c)
# %%
