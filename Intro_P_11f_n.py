#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: maryamramezaniziarani
"""

#Pandas;structured data
#%%
import pandas as pd
import numpy as np
#%% Series
g = pd.Series(['a', 'b', 'c'])
print(g)
type(g)
#%%
#Exercise1: create two Series from an array of numbers using numpy and check for the type
t= pd.Series(np.array([3,5,7,8]))
print(t)
type(t)
#%%
h= pd.Series(np.array([1,7,9]))
print(h)
type(h)
#%% Dataframe
i= pd.Series([7, 8, 9])
dataframe_dict = {'ID': g, 'value': h, 'condition': i}
dataframe = pd.DataFrame(dataframe_dict)
print(dataframe)

## Save DataFrame to CSV
dataframe.to_csv('neww_file.csv', index=False)

print(dataframe.shape)

#%%
import os
print(os.getcwd())

#%% Selecting Data

print(dataframe.columns)
print(dataframe['ID'])

#%%
#Exercise2:create a table (dataframe structure) from multiple series (including boolean
#and numiric) 
#and check the shape. 
#Select only the boolean column and save the indexed column to a CSV file.

series1 = pd.Series([1, 2, 3])
series2 = pd.Series([4, 5, 6])
series3 = pd.Series([False, True, False])

df_new = pd.DataFrame({'Val1': series1, 'Val2': series2, 'Val3': series3})
print(df_new)
print(df_new.shape)
df_new[['Val3']].to_csv('boolean.csv', index=True)

#%% #Descriptive statistics
print(df_new.info())
print(df_new.describe())
#%%
df = pd.read_csv('ind_inf.csv')
print(df.head())
print(df)
print(df.info())
#%%
#Methods (statistics)
#%%
df['Age'].mean()
#%%
df['Age'].median()
#%%
df['Age'].std()
#%% 
df['Age'][0:3].mean()
#%%Label-based
df.loc[:,'Age'].mean()
#%%Integer position-based
df.iloc[:2,2].mean()
#%%
#Exercise3:create a right-skewed dataset (using exponential distribution), load it into a Pandas DataFrame, 
#calculate its sum and variance and plot its histogram.

import matplotlib.pyplot as plt

## Create a right-skewed dataset using exponential distribution and load into a Pandas DataFrame
np.random.seed(42)  
data = np.random.exponential(size=1000) 
df = pd.DataFrame({'Value': data})  

print(df['Value'].sum())
print(df['Value'].var())

df['Value'].plot(kind='hist', bins=30, edgecolor='black', title='Histogram of Right-Skewed Data')
#plt.hist(df['Value'], bins=30, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid()
plt.show()

#%%
##Percentiles and Quartiles

# data
data = [12, 15, 18, 22, 25, 28, 32, 35, 38, 42, 45, 48, 52, 55, 58]

## Convert data to a pandas series
series = pd.Series(data)

## Calculate percentiles
percentiles = series.quantile([0.25, 0.5, 0.75])

print("25th Percentile (Q1):", percentiles[0.25])
print("50th Percentile (Q2 or Median):", percentiles[0.5])
print("75th Percentile (Q3):", percentiles[0.75])

## Calculate quartiles
quartiles = series.quantile([0, 0.25, 0.5, 0.75, 1])

print("\nMin:", quartiles[0])
print("25th Percentile (Q1):", quartiles[0.25])
print("50th Percentile (Q2 or Median):", quartiles[0.5])
print("75th Percentile (Q3):", quartiles[0.75])
print("Max:", quartiles[1])

#%%

data = [12, 15, 18, 22, 25, 28, 32, 35, 38, 42, 45, 48, 52, 55, 58]
series = pd.Series(data)

## Create a simple boxplot
plt.boxplot(series)
plt.title("Boxplot with Quartiles")
plt.ylabel("Values")
plt.xticks([1], ['My Data'])
plt.show()

#%%
#Exercise4:Consider a dataset with 1000 random observations. The data follows a standard 
#normal distribution, 
#and you are asked to find and plot the following:

#The 10th, 25th, 50th (median), 75th, and 90th percentiles.
#The first, second, and third quartiles.
#A box plot representing the quartiles.


## Generate a dataset with 1000 observations
np.random
complex_data = np.random.randn(1000)

## Create a DataFrame
df_complex = pd.DataFrame({'Values': complex_data})
print (df_complex)

## Calculate percentiles
percentiles = df_complex['Values'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])

print("10th Percentile:", percentiles[0.1])
print("25th Percentile:", percentiles[0.25])
print("50th Percentile (Median):", percentiles[0.5])
print("75th Percentile:", percentiles[0.75])
print("90th Percentile:", percentiles[0.9])

## Calculate quartiles
quartiles = df_complex['Values'].quantile([0, 0.25, 0.5, 0.75, 1])

print("\nMin:", quartiles[0])
print("25th Percentile (Q1):", quartiles[0.25])
print("50th Percentile (Q2 or Median):", quartiles[0.5])
print("75th Percentile (Q3):", quartiles[0.75])
print("Max:", quartiles[1])

## Create a box plot
plt.figure(figsize=(6, 4))  
plt.boxplot(df_complex['Values'])
plt.title("Box Plot with Quartiles")
plt.xticks([1], ['My Data'])
plt.ylabel("Values")


plt.show()

#%%
#Correlation
import pandas as pd

## data
data = {
    'Variable1': [1, 2, 3, 4, 5],
    'Variable2': [5, 4, 3, 2, 1],
    'Variable3': [2, 3, 1, 4, 5]
}

## Create a DataFrame
df = pd.DataFrame(data)

## Calculate correlation matrix
correlation_matrix = df.corr()

print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
plt.matshow(correlation_matrix, cmap='coolwarm')
plt.colorbar()
plt.title('Correlation Matrix')
plt.show()

#%%
#Exercise5:Create a DataFrame with two variables, A and B (The data follows a uniform distribution), and:

#Calculate the correlation coefficient between A and B.
#Visualize the correlation using a scatter plot.

## Create a DataFrame with variables A and B
data = {'A': np.random.rand(100), 'B': np.random.rand(100)}
df = pd.DataFrame(data)

## Calculate Correlation Coefficient
correlation_coefficient = df['A'].corr(df['B'])
print(f'Correlation Coefficient: {correlation_coefficient}')

## Visualize Correlation Using Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(df['A'], df['B'])
plt.title('Scatter Plot (A, B)')
plt.xlabel('A')
plt.ylabel('B')
plt.show()
#%%
#df.groupby 

## DataFrame
data = {'Student': ['Alice', 'Bob', 'Alice', 'Bob'],
        'Subject': ['Math', 'Math', 'Science', 'Science'],
        'Score': [85, 90, 88, 85]}

df = pd.DataFrame(data)

print(df)

grouped = df.groupby('Student')  

## Calculate the mean score for each student
mean_scores = grouped['Score'].mean()
print(mean_scores)

#%%
import scipy.stats as stats # statistical functions and tests

data = [1, 2, 2, 3, 3, 3, 4, 4, 5]

## Skewnes; sasymmetry of a distribution relative to its mean
skewness = stats.skew(data)

print("Skewness:", skewness)

#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#https://docs.sympy.org/latest/tutorials/intro-tutorial/intro.html
#https://ethanweed.github.io/pythonbook/03.01-descriptives.html
