#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 4 14:31:25 2024

@author: maryamramezaniziarani
"""

#Euclidean_algorithm
#%%
#Exercise1:Write a Python function, `euclidean_algorithm(a, b)`, that implements
#the Euclidean algorithm to find the greatest common divisor (GCD) of 
#two positive integers 'a' and 'b'. The function should return the GCD.
#%%
def euclidean_algorithm(a, b):
    if a <= 0 or b <= 0:
        raise ValueError("both numbers must be positive integers.")
    while b != 0:
        a, b = b, a % b
    return a

result = euclidean_algorithm(18, 48)
print("GCD:", result)

#%%
##List Comprehension [expression for item in iterable if condition]

#Exercise2:Create a list of even numbers from 0 to 9.

evens = [x for x in range(10) if x % 2 == 0]
print(evens)
#%%
#Exercise3:Write two functions to generate a list of squares of numbers from 0 to n-1:
#Function 1: generate_squares_comprehension(n) — This function should use list comprehension
#to create the list of squares.
#Function 2: generate_squares_loop(n) — This function should use a regular for loop 
#to create the list of squares.

def generate_squares_comprehension(n):
    return [x**2 for x in range(n)]
print(generate_squares_comprehension(5))
#%%
def generate_squares_loop(n):
    squares = []
    for x in range(n):
        squares.append(x**2)
    return squares
print(generate_squares_loop(5))
#%%
#Array and numpy
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
#Exercise4:Create a 3-dimensional array using np.arange() and
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
#Exercise5:Extract the value 15 from the z array above by indexing

z[1, 1, 2]
#%%
##array slice
#%%
z[1, 2, :]
#%%
y[:2, :]
#%%
y[1:, :]
#%%
z[::-1]
#%%
#Exercise6: Given a 3D NumPy array z, write a code that will reverse
#the order of the elements along the second axis
z[:, ::-1, :]

#%%
#Exercise7:Extract the 2nd layer, first and 2nd rows and all
#columns from the z array above by slice and indexing

z[1, 0:2, :]

#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
