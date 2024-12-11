#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 9 21:39:55 2024

@author: maryamramezaniziarani
"""
#%%
####### Random numbers
import numpy as np

##random.rand(); random float in the range [0.0, 1.0) from a uniform distribution
random_values = np.random.rand(2,2,4)

# Print the generated array
print(random_values)

#%%
#random.randn(a, b); axb matrix where each entry is drawn from a standard Gaussian distribution

random_matrix = np.random.randn(2,4)

# Print the generated matrix
print(random_matrix)

#%%
#Exercise1: Consider a matrix M generated using random.randn(a, b). Let M have dimensions 3×4.Each element M_ij
#in the matrix is drawn independently from a standard normal distribution (μ=0, σ=1). Calculate the following:

#a. The value of M_{21} (element in the second row, first column).
#b. The Z-score associated with the value M_{32} (element in the third row, second column).


## Generate a matrix M using random.randn(a, b)
matrix_M = np.random.randn(3, 4)

## Answer a: Value of M_{21} (element in the second row, first column)
M_21 = matrix_M[1, 0]

## Answer b: Z-score for M_{32} (element in the third row, second column)
M_32 = matrix_M[2, 1]
Z_score_M_32 = (M_32 - 0) / 1  

print("Matrix M:")
print(matrix_M)
print("\nAnswer a: Value of M_{21}:", M_21)
print("Answer b: Z-score for M_{32}:", Z_score_M_32)
#%%
#random.randint(a, b, size=n); Produces an array with n entries, each being an independently drawn 
#integer between a (inclusive) and b (exclusive) from a discrete uniform distribution

#np.random.seed(42)

roll_outcomes = np.random.randint(1, 7, size=2)

print(roll_outcomes)

#%%
# Exercise2: Write a code that generates a random integer number between 1 and 10 (inclusive) and asks the 
#user to guess the number. Provide feedback to the user if their guess is too high,too low, or correct.
#Continue this process until the user correctly guesses the number.

def guess_the_number():
    secret_number = np.random.randint(1, 11)
    attempts = 0

    print("Welcome to the Number Guessing Game!")
    
    # the first guess before the loop
    user_guess = int(input("Enter your guess (between 1 and 10): "))
    
    while user_guess != secret_number:
        if user_guess < secret_number:
            print("Too low! Try again.")
        elif user_guess > secret_number:
            print("Too high! Try again.")
        
      
        attempts += 1
        
        # Ask again inside the loop
        user_guess = int(input("Enter your guess (between 1 and 10): "))
    
    print(f"Congratulations! You guessed the number {secret_number} in {attempts} attempts.")

guess_the_number()

#%%
# np.random.choice, generates a random sample from a given 1-D array.
print(np.random.choice([10, 20, 30], size=5, replace= True, p=[0.2, 0.3, 0.5]))

#%%
#Exercise3: Creates a list of 6 student names, Assigns probabilities for selecting each student.
#Randomly selects 3 students without replacement using np.random.choice() based on the assigned probabilities.
#Prints the selected students.

# Step 1: List of student names
students = ["Alice", "Bob", "Charlie", "David", "Eve"]

# Step 2: Assign probabilities for selecting each student
probabilities = [0.1, 0.2, 0.4, 0.2, 0.1]

# Step 3: Randomly select 3 students without replacement
selected_students = np.random.choice(students, size=3, replace=False, p=probabilities)

# Step 4: Print the selected students
print("Selected students:", selected_students)

#%%
#vector and matrix norm; numpy.linalg.norm(m, ord, axis); m: the input for normalization; ord: The order of 
#the normalization; axis: it specifies the axis along which the vector norm of the matrix will be computed.
#L1 Norm
from numpy import array
from numpy.linalg import norm 
v = array([1,4,8])
l1 = norm(v,ord=1)
print(l1)

#%%
#Exercise4: Write a Python function named `calculate l1 norm' that takes a NumPy array as input and returns 
#the L1 norm of the array using the `numpy.linalg.norm` function. Use this function to find the L1 norm for the array `arr = np.array([-2, 5, -1, 0, 3])`.

def calculate_l1_norm(array):
    return np.linalg.norm(array, ord=1)

arr = np.array([-2, 5, -1, 0, 3])
l1_norm_result = calculate_l1_norm(arr)

print(f"the L1-norm of the array {arr} is: {l1_norm_result}")

#%%
#L2 Norm
v = array([1,4,8])
l2 = norm(v,2)
print(l2)
#%%
#Exercise5: Write a Python function named calculate_matrix_row_norms that takes a 2D NumPy array matrix as 
#input and returns a 1D array containing the L2 norm of each row in the matrix. Use numpy.linalg.norm to calculate the row norms.

def calculate_matrix_row_norms(matrix):
    row_norms = np.linalg.norm(matrix, ord=2, axis=1)
    return row_norms


sample_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_norms_result = calculate_matrix_row_norms(sample_matrix)

print(f"The row norms of the matrix are: {row_norms_result}")

#%%
#Matrix Frobenius Norm in NumPy

# Example matrix
A = np.array([[1, 2], [3, 4]])

# Compute Frobenius norm
norm_fro = np.linalg.norm(A, ord='fro')
print("Frobenius Norm:", norm_fro)

#%%
#Exercise6: Design a Python function named find_magical_matrix_norm that takes nine 
#integers as input, 
#constructs a 3x3 matrix, and calculates the Frobenius norm of the matrix using 
#numpy.linalg.norm. 
#The catch is that the integers you choose must satisfy a specific magical condition:
#the sum of each row, 
#each column, and both main diagonals should be the same.

#Your function should return the Frobenius norm of the magical matrix if the condition
#is met, 
#and an informative message otherwise.


def find_magical_matrix_norm(a, b, c, d, e, f, g, h, i):
    
    magical_matrix = np.array([[a, b, c], [d, e, f], [g, h, i]])

    # Check the condition
    magical_sum = np.sum(magical_matrix[0, :])  # Sum of the first row

    row_sums = np.sum(magical_matrix, axis=1)
    col_sums = np.sum(magical_matrix, axis=0)
    main_diag_sum = np.trace(magical_matrix)
    anti_diag_sum = np.trace(np.flipud(magical_matrix))

    if all(row_sum == magical_sum for row_sum in row_sums) and \
       all(col_sum == magical_sum for col_sum in col_sums) and \
       main_diag_sum == magical_sum and \
       anti_diag_sum == magical_sum:
        # If the condition is met, calculate the Frobenius norm
        frobenius_norm = np.linalg.norm(magical_matrix, 'fro')
        return frobenius_norm
    else:
        return "The chosen integers do not create a magical matrix."


result = find_magical_matrix_norm(4, 9, 2, 3, 5, 7, 8, 1, 6)
print(result)
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)































