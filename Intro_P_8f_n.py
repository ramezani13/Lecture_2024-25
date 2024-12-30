#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:15:29 2024

@author: maryamramezaniziarani
"""

#%%
#try-except blocks for error handling 
#%%
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Division by zero is not allowed.")
        return None

# Example:
numerator = 10
denominator = 0

result = safe_divide(numerator, denominator)

if result is not None:
    print(f"The result of {numerator} / {denominator} is: {result}")
else:
    print("Error occurred during division.")
#%%
#Exercise1: Write a Python function called get_element(my_list, index) that takes a list my_list
#and an index as input. The function should attempt to access the element at the
#specified index in the list using a try-except block. If the index is valid, it should return
#the element; otherwise, it should print an error message and return None.

def get_element(my_list, index):
    try:
        element = my_list[index]
        return element

    except Exception as e:
        print(f"Error: {e}")
        return None

## Example:
my_list = [10, 20, 30, 40, 50]

index_to_access = 7

element = get_element(my_list, index_to_access)

if element is not None:
    print(f"The element at index {index_to_access} is: {element}")
else:
    print(f"Error: Cannot access element at index {index_to_access}.")

#%%
#Scope 
#%%
# global scope variable
global_number = 42

def example_function():
    # local scope variable
    local_number = 7
    print("Inside the function:")
    print("Global variable:", global_number)  # global variable within the function
    print("Local variable:", local_number)    # local variable within the function

example_function()

print("Outside the function:")
print("Global variable:", global_number)
#print("Local variable:", local_number)
#%% 
#nested functions
def outer_function():
    outer_variable = "I am outer"

    def inner_function():
        nonlocal outer_variable
        inner_variable = "I am inner"
        
        # Accessing and modifying the outer_variable from the enclosing scope
        print("Inside inner function:")
        print("Before modification - outer_variable:", outer_variable)
        outer_variable = "Modified outer from inner"
        print("After modification - outer_variable:", outer_variable)
        print("inner_variable:", inner_variable)

    inner_function()

    # Accessing outer_variable outside inner_function
    print("Outside inner function - outer_variable:", outer_variable)

# Call the outer function
outer_function()
#%%
#Exercise2:
def outer_function(x):
    outer_variable = x

    def inner_function(y):
        # Challenge 1: Use nonlocal to modify outer_variable
        inner_variable = y

        # Challenge 2: Try to access x from outer_function inside inner_function
        
    inner_function(20)

    # Challenge 3: Try to access inner_variable outside inner_function
    

# Challenge 4: Access outer_variable outside outer_function

outer_function(10)
#%%
def outer_function(x):
    outer_variable = x

    def inner_function(y):
        nonlocal outer_variable  # Challenge 1: Use nonlocal to modify outer_variable
        inner_variable = y

        # Challenge 2: Try to access x from outer_function inside inner_function
        print("Accessing x from outer_function inside inner_function:", x)
        print("Inside inner function:")
        print("outer_variable:", outer_variable)
        print("iner_variable:", inner_variable)

    inner_function(20)

    # Challenge 3: Try to access inner_variable outside inner_function
    #print("Trying to access inner_variable outside inner function:", inner_variable)

# Challenge 4: Access outer_variable outside outer_function
#print("Accessing outer_variable outside outer function:", outer_variable)

# Call the outer function with an argument
outer_function(10)
#%%
#Functions as First-Class Objects
#%%
#Assigning functions to variables:
    
def square(x):
    return x * x

def cube(x):
    return x * x * x

function_variable = square
print(function_variable(5))

#Passing functions as arguments:
    
def apply_operation(func, y):
    return func(y)

result = apply_operation(cube, 3)
print(result)  

#Returning functions from a function:

def get_function(power):
    if power == 2:
        return square
    else:
        return cube
result = get_function(4)
print(result(4))  

#%%
##Exercise3:
    
#Write a Python function called manipulate_numbers that takes two numbers, x and y, 
#and a function manipulation_func as arguments. The function should apply the provided 
#manipulation_func to the two numbers and return the result. Additionally, 
#demonstrate the use of this function with various manipulation functions.

#manipulation functions:

#addition, subtraction, multiplication, power

#Provide examples of using the manipulate_numbers function with each manipulation function. 
  

# Manipulation function: addition
def addition(x, y):
    return x + y

# Manipulation function: subtraction
def subtraction(x, y):
    return x - y

# Manipulation function: multiplication
def multiplication(x, y):
    return x * y

# Manipulation function: power
def power(x, y):
    return x ** y

def manipulate_numbers(x, y, manipulation_func):
    return manipulation_func(x, y)

# Using the manipulate_numbers function
result1 = manipulate_numbers(5, 3, addition)        
result2 = manipulate_numbers(7, 4, subtraction)      
result3 = manipulate_numbers(2, 6, multiplication)    
result4 = manipulate_numbers(2, 3, power)  

#%% Recap
#Exercise1: #You are analyzing the grades of students in a class. The grades are stored in a NumPy array,
#where each row represents a student, and each column represents a subject. Perform the following
#operations step by step: hint: np.where, np.all, np.cumsum, np.argmin

# Step 1: Mean grades for each subject
# Step 2: Identify students who scored above the subject average in all subjects
# Step 3: Compute cumulative grades for each student
# Step 4: Find the student with the lowest total grade

#numpy.where() function to select elements from a numpy array, based on a condition.
#np.all tests whether all elements along a specified axis evaluate to True.
#np.argmin returns the index of the smallest element in an array along a specified axis


import numpy as np
import matplotlib.pyplot as plt

# Example grades: rows = students, columns = subjects
grades = np.array([
    [85, 90, 90],   # Student 1
    [88, 76, 95],   # Student 2
    [92, 88, 82],   # Student 3
    [70, 65, 85]    # Student 4
])

# Step 1: Mean grades for each subject
subject_means = np.mean(grades, axis=0)
print("Mean grades for each subject:", subject_means)

# Step 2: Identify students who scored above the subject average in all subjects
above_average = np.where(np.all(grades > subject_means, axis=1))[0]
print("Students scoring above the average in all subjects:", above_average)

# Step 3: Compute cumulative grades for each student
cumulative_grades = np.cumsum(grades, axis=1)
print("Cumulative grades for each student:\n", cumulative_grades)

# Step 4: Find the student with the lowest total grade
total_grades = np.sum(grades, axis=1)
print(total_grades)
lowest_student = np.argmin(total_grades)
print(f"Student with the lowest total grade: Student {lowest_student + 1}")

#%%
#Exercise2: Create a Python script using NumPy and Matplotlib to fit a quadratic curve to 
#the given data points
#and plot the results. The data points are:
    
# Given data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 6, 9, 15])

# Fit a quadratic curve to the data
coefficients = np.polyfit(x, y, 2)
print(coefficients)



# Use polyval to calculate y values for the curve
y_curve = np.polyval(coefficients, x)
print(y_curve)

# Plot data points as blue dots
plt.scatter(x, y, color='blue', label='Data Points')

# Plot the quadratic curve as a red curve
plt.plot(x, y_curve, color='red', label='Quadratic Trendline')

# Set axis labels
plt.xlabel('X')
plt.ylabel('Y')

# Set the legend
plt.legend()

# Show the plot
plt.show()
#%%
#Exercise3: Consider a power-law function (y = cx^alpha), where (c = 2) and (alpha = 1.5). Create a plot of 
#this power-law function for (x) values in the range ([1, 10]), # Given parameters c = 2 and alpha = 1.5.
#Use NumPy and Matplotlib to generate the plot.

# Define the power-law function
def power_law(x, c, alpha):
    return c * x ** alpha

# Given parameters
c = 2
alpha = 1.5

# Generate x values in the range [1, 10]
x_values = np.linspace(1, 10, 20)
print(x_values)

# Calculate corresponding y values using the power-law function
y_values = power_law(x_values, c, alpha)
print(y_values)

# Create a plot
plt.plot(x_values, y_values, label='Power Law')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Power Law Plot')
plt.legend()
plt.grid(True)
plt.show()
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)