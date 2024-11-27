#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 26 13:21:23 2024

@author: maryamramezaniziarani
"""

#%% #complex Boolean masks using logical operators
#%%
#& (AND): both conditions are true 
#| (OR): at least a condition is true
#~ (NOT): negates the condition
#Rules for Using Logical Operators:
#Parentheses are required around each condition because logical 
#operators have a lower precedence than comparison operators.
#The conditions are applied element-wise to the array.
#%%
import numpy as np

x = np.array([5, 11, 15, 20, 25])
mask = (x > 11) & (x < 25)  ## Select elements between 11 and 25

#print (mask)
print(x[mask])

mask = (x == 11) | (x == 25) ## Select elements equal to 10 or 25
print(x[mask]) 

mask = ~(x > 15)  ## Select elements less than or equal to 15
print(x[mask])

#%%
#Exercise 1: given a numpy array scores representing test scores, 
#create a new array that includes:

#Scores greater than 80 AND less than 90, OR Scores equal to 100 
#and set all other scores to 0 in the original array.

#scores = np.array([50, 85, 95, 100, 80, 89, 90])


scores = np.array([50, 85, 95, 100, 80, 89, 90])

## Logical condition: greater than 80 and less than 90, or equal to 100
mask = ~(((scores > 80) & (scores < 90)) | (scores == 100))

## Set all scores that match the mask to 0
scores[mask] = 0

print(scores) 
#%%
##Array Summarizing
#Summarization involves calculating aggregated values like the sum, mean
#, maximum, minimum, etc., 
#across the entire array or along specific axes.

arr = np.array([[1, 2], [3, 4]])
total_sum = arr.sum()
print(total_sum)
row_sums = arr.sum(axis=1)
print(arr)
print(row_sums)

row_prod = arr.prod(axis=1)
print(row_prod)

#%%
mean = arr.mean(axis=0) 
print(mean)
#%%
minimum = arr.min(axis=0)
print(minimum)
#%%
arr.max()
#%%
arr.std()
#%%
arr.var()
#%%
np.median(arr)
#%%
#Exercise 2: 
#You are given a 3D numpy array:
arr_test = np.array([
        [[2, 4], [6, 8]],
        [[1, 3], [5, 7]],
        [[0, 2], [4, 6]]
])

print (arr_test)
#Compute the sum of all elements in the array.
#Calculate the mean along axis=0.
#collapse the rows and calculate the max for each slice.
#collapse the columns and calculate the standard deviation for all slice. 


# 1. Total sum of all elements
arr_test_sum = arr_test.sum()
print(arr_test_sum) 

# 2. Mean along axis=0 
arr_test_mean = arr_test.mean(axis=0)  
print(arr_test_mean)

# 3. Maximum value along axis=1 
arr_test_max = arr_test.max(axis=1)


# [[6, 8],  # layer 1
#  [5, 7],  # layer 2
#  [4, 6]]  # layer 3

# 4.collapse the columns and slices
arr_test_std = arr_test.std(axis=(0,2))
print("show me", arr_test_std)

#%%
#Exercise 3: 
#You are given a 2D NumPy array:
    
arr_test2 = np.array([
    [12, 15, 20],
    [10, 25, 30],
    [18, 21, 24]
])

#Find the sum of all even numbers in the array.

# 1. Sum of all even numbers
#mask = arr_test2 % 2 == 0
even_sum = arr_test2[arr_test2 % 2 == 0].sum()
print(even_sum)
#%%
#Exercise4: a) Write a Python program to find the median of a given 
#array of numbers 
#using conditional statements (if) 
#b) Write a Python program to find the 
#median of a given array
#of numbers using summarizing

m=[2,3,5,7,8,9,11]

#using if condition

array_len = len(m)

if array_len % 2 == 0:
    median = (m[int(array_len/2)] + m[int(array_len/2 -1)])/2
else:
    median = m[int(array_len/2)]

print(median)

#b) np.median(m)

#%%
#Diagonal Operations and Symmetry
#%% Use np.diagonal() to get the diagonal elements of a matrix:
matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])

diagonal_elements = np.diagonal(matrix)
print("Diagonal elements:", diagonal_elements) 
#%% Use np.trace() to directly compute the sum of diagonal elements:
diagonal_sum = np.trace(matrix)
print("Sum of diagonal elements:", diagonal_sum)  
#%%
#The transpose of a matrix interchanges its rows and columns:
transpose = matrix.T
print("Transpose of matrix :", transpose)
#%% A symmetric matrix is a square matrix that is equal to its transpose. a[i,j]=a[j,i]
def is_symmetric(matrix):
    return np.allclose(matrix, matrix.T)

symmetric_matrix = np.array([[1, 2, 3],
                              [2, 5, 6],
                              [3, 6, 9]])
print("Is symmetric?", is_symmetric(symmetric_matrix)) 

#%%
#Exercise5: Consider the following NumPy array:

matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])

#a) Write a Python program to find the sum of the diagonal elements of a given 
#square matrix without using the np.trace function and without summarizing. Use 
#loops for iteration.

#b) Implement a function that takes a square matrix as input and returns True if 
#the matrix is symmetric and False otherwise. Avoid using summarizing NumPy 
#operations and use loops for element comparison.

#c) Write a program to calculate the product of each column in the matrix without 
#using np.prod and without summarizing. Use nested loops for the calculation.

#a)# Solution for part (a) without summarizing

matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])


diagonal_sum = 0
for i in range(len(matrix)):
    diagonal_sum += matrix[i, i]

print("Sum of diagonal elements without summarizing:", diagonal_sum)

#b) # Solution for part (b) without summarizing

def is_symmetric(matrix):
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] != matrix[j, i]:
                return False
    return True

matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])

symmetric_result = is_symmetric(matrix)
print("Is the matrix symmetric?", symmetric_result)


#c) # Solution for part (c) without summarizing

matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])


rows, cols = matrix.shape
column_products = np.ones(cols)
print(column_products)

for j in range(cols):
    for i in range(rows):
        column_products[j] *= matrix[i, j]

print("Product of each column without summarizing:", column_products)

#%%
# a)# Solution for part (a)

matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])


diagonal_sum = np.trace(matrix)
print("Sum of diagonal elements:", diagonal_sum)

# b) # Solution for part (b)


def is_symmetric(matrix):
    return np.allclose(matrix, matrix.T)

matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])

symmetric_result = is_symmetric(matrix)
print("Is the matrix symmetric?", symmetric_result)

# c) # Solution for part (c)


matrix = np.array([[5, 3, 8],
                   [1, 6, 2],
                   [7, 4, 9]])


column_products = np.prod(matrix, axis=0)
print("Product of each column:", column_products)
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
