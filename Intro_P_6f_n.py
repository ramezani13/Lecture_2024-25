#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: maryamramezaniziarani
"""
#%%
import os
print(os.getcwd())
#%% ##reading the data from the file, saving the data to the file
import numpy as np
x = np.array([5, 4, -7, 11])
print(x)
#%%
y = np.array([[4, 5, 0], [3, 7, 9], [2, 1, 2]])
print(y)
#%%
np.savetxt("y_test.txt", y, delimiter=",", header="name,id,region", fmt="%d")
#%%
b = np.genfromtxt('testn.txt', delimiter=';')
print(b) 
#%%
b = np.genfromtxt('testn.txt', delimiter=';', skip_header=1)
print(b)
#%%
#Exercise1: Create a 2D array and use the np.savetxt() function to save the data array to a text 
#file named "mydata.txt" with a comma delimiter and a header line Col1,Col2,Col3,Col4

data = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])

np.savetxt("mydata.txt", data, delimiter=",", header="Col1,Col2,Col3,Col4", fmt="%d")

print("Data saved to mydata.txt file.")

#%% ## ploting an array
#%%
import matplotlib.pyplot as plt
#%%
plt.plot(y[1, :]);
#%%
plt.plot(y[:, 1]);
#%%
plt.plot(y);
#%%
plt.plot(y, color='red');
#%%
#Exercise2: Plot each column of the array as a separate line.
#Assign the following colors to each line: ['red', 'blue', 'green'], using for loop.


## colors for each line
colors = ['red', 'blue', 'green']

for i in range(y.shape[1]):  ## loop over columns
    plt.plot(y[:, i], color=colors[i], label=f"Column {i}")

plt.legend()
plt.title("Plot with Specified Colors")
plt.show()

#%%
plt.plot(y[:, 0], color='red', label='red')
plt.plot(y[:, 1], color='green', label='green')
plt.plot(y[:, 2], color='blue', label='blue')
plt.legend();
plt.title('Line Plot for Each Column in y')
plt.xlabel('X-axis (Assuming Index)')
plt.ylabel('Y-axis (Values in Each Column)')
plt.show()
#%%
#Exercise3: Given an 2D array of numbers, plot a graph that displays the sum of 
#each element in the column.

arr = [[1,2,3],
       [4,5,6],
      [7,8,9]]

column_sums= np.sum(arr, axis=0)
print(column_sums)
plt.plot(column_sums)
plt.title('Sum of elements in each column')
plt.xlabel('column')
plt.ylabel('Sum of elements')
plt.show()
#%%
##Histograms
y = np.array([[4, 5, 0], [3, 7, 9], [2, 1, 2]])
h = y.flatten()
print(h)
plt.hist(h);
#%%
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

plt.hist(data, bins=[0,2,4,6,8,10])
plt.title('My Histogram')
plt.xlabel('Interval')
plt.ylabel('Frequency')
plt.show()
#%%
##Line Plots

x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

plt.plot(x, y)
plt.show()
#%%
##linear Regression Fit and trend line

x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

coeficients = np.polyfit(x, y, 1)
print(coeficients)
trend_line = np.polyval(coeficients, x)


## plot the orignal data points
plt.plot(x, y, '*', label='Data Points')


# plot the trend line
plt.plot(x, trend_line, label='Trend Line', color='red')

## add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Trend Line')
plt.legend()

## show the plot
plt.show()
#%%
##Scatter Plots

x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

plt.scatter(x, y)
plt.show()
#%%
##Bar Plots, categorical data

categories = ['A', 'B', 'C']
values = [4, 7, 2]

plt.bar(categories, values)
plt.show()

#%%
#Exercise4:
#Consider a scenario where the relationship between two variables, `x` and `y`, is described by a quadratic 
#function: y = ax^2 + bx + c

#a).  Write a Python function, `generate_quadratic_data(a, b, c, num_points)`, that takes coefficients a, b, and
#c, and the number of data points `num_points`, and generates synthetic data according to the quadratic function.
#Assume x values in the range from -10 to 10, using x = np.linspace(-10, 10, num_points).
    
#b). Use the function to generate a dataset with a=1, b=-2, c=1, and num_points=100


#c). Write another function, plot_quadratic_scatter(x, y), that takes the generated x and y values and creates a 
#scatter plot.

##a
def generate_quadratic_data(a, b, c, num_points):
    x = np.linspace(-10, 10, num_points)
    y = a * x**2 + b * x + c
    return x, y


## Generate quadratic data
a, b, c = 1, -2, 1
num_points = 100
x_data, y_data = generate_quadratic_data(a, b, c, num_points)

## Plot the scatter plot
def plot_quadratic_scatter(x, y):
    plt.scatter(x, y, label='Generated Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Quadratic Data Scatter Plot')
    plt.legend()
    plt.show()
plot_quadratic_scatter(x_data, y_data)
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)






















