#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 19 14:33:11 2024

@author: maryamramezaniziarani
"""

#%%
#Array and numpy
#Array creation
#%%
import numpy as np
x = np.array([5, 4, -7, 11])
print(x)
#%%
y = np.array([[4, 5, 0], [3, 7, 9], [2, 1, 2]])
print(y)
#%%
z = np.array([[[1,2,3], [4,5,6], [7,8,9]], [[10,11,12], [13,14,15], [16,17,18]]])
print(z)
#%%
b = np.arange(10).reshape(2, 5) 
print(b)
#%%
type(b)
#%%
b.shape
#%%
b.size
#%%
b.ndim
#%%
#Exercise 1:Create a 3-dimensional array using np.arange() and
#reshape and check for shape, size, dimension

c = np.arange(18).reshape((3, 3, 2))
print(c)
#%%
c.shape
#%%
c.size
#%%
c.ndim
#%%
l = np.linspace(1, 10, 7)
print (l)
#%%
#Exercise2: Generate 4 equally spaced numbers from 0 to 11 (including 11) using np.linspace. 
#Reshape the resulting array into a 2x2 matrix and print the reshaped array.

array_lin = np.linspace(0, 11, 4)
print("array_lin:", array_lin)

array_lin_re = array_lin.reshape(2, 2)
print("array_lin_re", array_lin_re)

#%%
#Exercise3:Create a 3D array with np.linspace()

array_3d = np.linspace(1, 27, 27).reshape((3, 3, 3))
print(array_3d)

#%%
arr_ones = np.ones((2,2,4)) 
print(arr_ones)
#%%
arr_zeros = np.zeros((2, 3), dtype=int) 
print(arr_zeros)
#%%
#Exercise4:Create a 6x6 matrix initialized with zeros. Set the diagonal elements to 1, 
#the elements diretly above the diagonal to 2, and the elements directly below the diagonal to 3. 
#Print the resulting matrix.

#[[1 2 0 0 0 0]
#[3 1 2 0 0 0]
#[0 3 1 2 0 0]
#[0 0 3 1 2 0]
#[0 0 0 3 1 2]
#[0 0 0 0 3 1]]


matrix_six = np.zeros((6, 6), dtype=int)


for i in range(6):
    matrix_six[i, i] = 1

for i in range(5):  ## up to second to last row
    matrix_six[i, i + 1] = 2

for i in range(1, 6):  ## from the second row
    matrix_six[i, i - 1] = 3

print(f" my resulting matrix:{matrix_six}")
#%%
## array indexing and slice
#%%
#Exercise5:Extract the value 15 from the z array above by indexing
z[1, 1, 2]
#%%
z[1, 2, :]
#%%
y[:2, :]
#%%
y[1:, :]
#%%
z[::-1]
#%%
#Exercise6: given a 3D numpy array z, write a code that will reverse
#the order of the elements along the second axis
z[:, ::-1, :]

#%%
#Exercise7:Extract the 2nd layer, first and 2nd rows and all
#columns from the z array above by slice and indexing
z[1, 0:2, :]

#%% np.concatenate() 

arrx = np.array([3, 7, 2])
arry = np.array([1, 5, 6])

arr_new = np.concatenate((arrx, arry))

print(arr_new)

#%%
#Exercise8: given two 3D numpy arrays arr1 and arr2, write a line of code to concatenate 
#these arrays along the second axis (rows). What will be the shape of the resulting array?

arr1 = np.array([
    [[1, 2], [3, 4]], 
    [[5, 6], [7, 8]]
])

arr2 = np.array([
    [[9, 10], [11, 12]], 
    [[13, 14], [15, 16]]
])

#print(arr1)
#print(arr2)
result = np.concatenate((arr1, arr2), axis=1)
print (result)
print(result.shape)

#%% Boolean Indexing

## students scores
scores = np.array([85, 60, 95, 45, 70])

## boolean list 
pass_fail = [True, False, True, False, True]

## boolean indexing 
passed_scores = scores[pass_fail]

print("Original scores:", scores)
print("Pass/Fail filter:", pass_fail)
print("Scores of students Who pased:", passed_scores)

#%%
# Exercise9: Given a numpy array numbers with the values [10, 15, 20, 25, 30, 35, 42], 
#write a code to mask out only the elements that are divisible by 5. 
#Use list Comprehension
#or
#Use a for loop to create a boolean mask list named is_divisible_by_5, and then 
#use that mask to create a new array named masked_numbers.

numbers = np.array([10, 15, 20, 25, 30, 35, 42])

## create the boolean mask using list comprehension
is_divisible_by_5 = [number % 5 == 0 for number in numbers]
print(is_divisible_by_5)

## using the boolean mask to filter the numbers
masked_numbers = numbers[is_divisible_by_5]

print("Boolean filter array:", is_divisible_by_5)
print("Filtered numbers array:", masked_numbers)

#%%
numbers = np.array([10, 15, 20, 25, 30, 35, 42])

is_divisible_by_5 = []

for number in numbers:
    ## If the number is divisible by 5, the value is True, otherwise False
    if number % 5 == 0:
        is_divisible_by_5.append(True)
    else:
        is_divisible_by_5.append(False)

masked_numbers = numbers[is_divisible_by_5]

print("Boolean filter array:", is_divisible_by_5)
print("Filtered numbers array:", masked_numbers)
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org