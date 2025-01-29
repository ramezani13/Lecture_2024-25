#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 08:13:58 2025

@author: maryamramezaniziarani
"""
#%%
#1. A number is a palindrome if the i-th decimal digit from the front coincides with the 
#i-th decimal digit from the back. E.g., 18781 is a palindrome, whereas 84736 is not. 
#Write a function is_palindrome(n) that returns True if 
#n is a palindrome, and False otherwise.

def is_palindrome(n):
    n_str = str(n)
    
    return n_str == n_str[::-1]  
print(is_palindrome(121)) 
#%%
#2. Write a Python function sum_of_squares(n) that calculates the sum of squares of all odd numbers up to and including n.
#The function should return the sum as a float if the input is valid, or return -1 if the input is invalid. 
#Additionally: If the input is not a positive whole number, return -1 to indicate an error If the input 
#is greater than 1000, print a warning message but still perform the calculation.

def sum_of_squares(n):
    if type(n) != int or n <= 0:
        return -1 

    if n > 1000:
        print("Warning: Large input value. Calculation might take longer.")  

    total = 0  
    i = 1  
    while i <= n:
        if i % 2 != 0:  
            total = total + (i * i)  
        i = i + 1  

    return float(total)  
print(sum_of_squares(5)) 
#%%
#3. Write a Python function count_odd_digits(n) that takes a non-negative integer n and returns the count of 
#odd digits in that number. Your function cannot use multiplication, modulo, or division operators. 
#You cannot convert the number to a string or use any string operations. 
#If the input is negative, return -1 to indicate an error. Although it's an unconventional way to do 
#this task with the above-mentioned constraints, the goal of the task is to encourage you to think more creatively.

def count_odd_digits(n):
    if n < 0:
        return -1  
    
    count = 0  
    while n > 0:  
        digit = n  
        
        while digit > 9:
            digit = digit - 10
       
        if digit == 1 or digit == 3 or digit == 5 or digit == 7 or digit == 9:
            count = count + 1  
        
        n = n // 10

    return count  
print(count_odd_digits(125)) 
#%%
#4. Write a Python function, is_odd(k), that takes an integer value and returns True
#if k is an odd number, and False otherwise. However, your function can not use the multiplication, 
#modulo, or division operators.

def is_odd(k):
    word = str(k)  
    if word[-1] not in "02468":  
        return True  
    else:
        return False  
print(is_odd(46))
#%%
#5. Write a function gcd(a, b) that implements the extended Euclidean algorithm. 
#your function should return a tuple (d, x, y) where:
#d is the greatest common divisor of a and b
#x and y are the coefficients such that ax + by = d (Bézout's identity) For example, 
#if a=13 and b=19, your function should return a tuple(1, x, y) where 13x + 19y = 1.

def gcd(a, b):
    if b == 0:
        return a, 1, 0
    
    t, x1, y1 = gcd(b, a % b)
    
    x = y1
    y = x1 - (a // b) * y1
    
    return t, x, y

print(gcd(12, 8))
#%%
#6. Write a function get_reversed_list(L) that takes a list L and returns the list in reverse order,
#along with the number of swaps performed during the reversal process. Your function should return
#a tuple (reversed_list, num_swaps).

def get_reversed_list(L):
    reversed_list = L.copy()
    num_swaps = 0
    
    for i in range(len(reversed_list) // 2):
        reversed_list[i], reversed_list[len(reversed_list) - 1 - i] = reversed_list[len(reversed_list) - 1 - i], reversed_list[i]
        num_swaps += 1
    
    return reversed_list, num_swaps

L = [1,2,3,4]
print (get_reversed_list(L))
#%%
#7. Write a function powers_of_2(n) which returns a list containing the numbers 
#$2^0, \ldots, 2^{n-1}$.Your code must not contain more than 55 characters. Use list comprehensions.

def powers_of_2(n):
  return [2**i for i in range(n)]

print (powers_of_2(4))

#%%
#8. Modify the powers_of_2(n) function to also return the sum of the generated powers of 2.
#The function should now return a tuple (powers_of_2, sum_of_powers).

def powers_of_2(n):
    powers = [2**i for i in range(n)]
    return powers, sum(powers)

print (powers_of_2(4))

#%%
#9. Write a Python function make_change(charged,given) that "makes change". 
#The function should take two integers as arguments: the first represents the monetary 
#amount charged, and the second is the amount given, both in Euro cents. The function 
#should return a Python dictionary containing the Euro bills or coins to give back as 
#change between the amount given and the amount charged. Design your program to return 
#as few bills and coins as possible.
#For example, make_change(6705, 10000) should 
#return {20.0: 1, 10.0: 1, 2.0: 1, 0.5: 1, 0.2: 2, 0.05: 1}.
#If the provided inputs are not integers or if the amount given is less than the amount 
#charged, the function should raise a ValueError with an appropriate error messages;
#"Both 'charged' and 'given' should be integers." and "Too little money given".


def make_change(charged, given):
    if type(charged) != int or type(given) != int:
        raise ValueError("Both 'charged' and 'given' should be integers.")  

    
    if given < charged:
        raise ValueError("Too little money given")  

    
    denominations = [500 * 100,  # 500 Euro bill in cents
                     200 * 100,  
                     100 * 100,  
                     50 * 100,   
                     20 * 100,   
                     10 * 100,   
                     5 * 100,    
                     2 * 100,    
                     1 * 100,    
                     50,         
                     20,         
                     10,         
                     5,          
                     2,          
                     1]          

     
    money_used = {}

    
    remaining = given - charged  

    
    for amount in denominations:
        if remaining >= amount:  
            factor = remaining // amount  
            money_used[amount / 100] = factor  
            remaining -= factor * amount  

    
    assert remaining == 0  

    ## our dictionary 
    return money_used

charged = 6705  ## €67.05 in cents
given = 10000   ## €100.00 in cents

result3 = make_change(charged, given)
print(result3)
#%%
#10. (From GTG, C-1.14.) Write a short Python function has_odd_product(L) that takes a 
#list of integer values L and determines if there is a distinct pair of numbers in 
#the sequence whose product is odd.

def has_odd_product(L):
    s = set()
    for i in L:
        if i % 2 == 1:  
            if i not in s:  
                if len(s) != 0:  
                    return True
                
                s.add(i)  
    return False  

L = [2, 4, 6, 8, 3, 3]
result = has_odd_product(L)
print(result)
#%%
#11. (From GTG, C-1.15.) Write a Python function are_distinct(L) that takes a list of 
#numbers L and determines if all the numbers are different from each other
#(that is, they are distinct).

def are_distinct(L):
    s = set()  
    for i in L:  
        if i in s:  
            return False  
        else:
            s.add(i)  
    return True  

L = [2, 4, 6, 8, 3, 3]
result2 = are_distinct(L)
print(result2)

#or

def are_distinct(L):
    return len(L) == len(set(L))
#%%
#12. Write a function named `process_array` that processes array elements 
#based on specific conditions. 
#The function should take three parameters:
#- input_array: A NumPy array of integers
#- lower_bound: An integer representing the lower threshold
#- upper_bound: An integer representing the upper threshold

#The function should return a new NumPy array where:
#a) Elements less than lower_bound are cubed
#b) Elements between lower_bound and upper_bound (inclusive) that are divisible by 3 are doubled
#c) All other elements remain unchanged

import numpy as np

def process_array(input_array, lower_bound, upper_bound):
    
    result_array = np.copy(input_array)
    
    result_array[input_array < lower_bound] = input_array[input_array < lower_bound] ** 3

    mask_between_bounds = (input_array >= lower_bound) & (input_array <= upper_bound) & (input_array % 3 == 0)
    result_array[mask_between_bounds] = input_array[mask_between_bounds] * 2

    
    return result_array

input_array = np.array([1, 6, 9, 3, 15, 8])
lower_bound = 5
upper_bound = 10

result = process_array(input_array, lower_bound, upper_bound)
print("Processed array:", result)
#%%
#13. Write a function `get_primes_up_to(n)` which takes as input a natural number
#$n$ and returns a numpy array containing all the prime numbers smaller or equal to $n$.

def get_primes_up_to(n):
    is_prime = np.ones(n+1, dtype=bool)  
    is_prime[0:2] = False  

    
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:  
            is_prime[i*i : n+1 : i] = False  
            is_prime[i*i : n+1 : i] = False  
    primes = np.nonzero(is_prime)[0]  
    return primes                

n = 20
primes = get_primes_up_to(n)
print(f"Prime numbers up to {n}: {primes}")
#%% 
#14. Write a function `get_sum_of_composites_and_primes_squared_up_to(n)` that takes as 
#input a natural number $n$ and returns the sum of all composite (non-prime) numbers smaller
#or equal to $n$ plus the sum of squared primes smaller or equal to $n$. 
#Consider importing the `get_primes_up_to(n)` function from Problem2. 
#Since the prime numbers follow no known regular pattern, 
#a basic indexing solution to this problem is not easy. 
#Indexing by an array instead of by slicing can prove useful.

from get_primes_up_to import get_primes_up_to  

def get_sum_of_composites_and_primes_squared_up_to(n):
    primes = get_primes_up_to(n)  
    sum_of_squared_primes = np.sum(primes**2)  
    
    composites = np.setdiff1d(np.arange(1, n+1), primes)  
    sum_of_composites = np.sum(composites) 
    
    return sum_of_composites + sum_of_squared_primes  

#%%
#15. Design a NumPy function analyze_complex_array(arr) that performs multi-step analysis on a square 
#numerical array, with the following transformations:
#a) For each array element arr[i,j]:
#- If the element is divisible by 2 and exceeds its column's average value, then: replace the element with its row's total sum.
#- If no elements in a row meet the above criteria, then: replace the entire row with its cumulative (running) sum.
#b) Manually extract main and anti-diagonals without using np.diag(). Compute pairwise differences 
#between the main diagonal elements arr[i,i] and the corresponding anti-diagonal elements arr[i,n-1-i].
#Calculate the absolute differences, and return the total sum of these differences.
#c) Sort each row individually and identify the lexicographically smallest row and return the index of the smallest row and
#the sorted version of that row.

#Input Constraints:
#Must be a square NumPy numerical array.
#Minimum dimensions: 4x4.
#Raises ValueError for:
#Non-square arrays.
#Arrays smaller than 4x4.

import numpy as np

def analyze_complex_array(arr):

    
    if not isinstance(arr, np.ndarray):  
        raise ValueError("Input must be a NumPy array")
    
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:  
        raise ValueError("Input must be a square array")
    
    if arr.shape[0] < 4:  
        raise ValueError("Array must be at least 4x4")
    
    n = arr.shape[0]  
    result_arr = arr.copy()  

    
    column_means = np.mean(arr, axis=0)  
    for i in range(n):
        row_condition = (arr[i] % 2 == 0) & (arr[i] > column_means)
        if np.any(row_condition):  
            result_arr[i] = np.where(row_condition, np.sum(arr[i]), arr[i])
        else:  
            result_arr[i] = np.cumsum(arr[i])
    
    
    main_diag = np.array([arr[i, i] for i in range(n)])
    anti_diag = np.array([arr[i, n - 1 - i] for i in range(n)])
    
    
    diagonal_differences = np.abs(main_diag - anti_diag)
    diagonal_sum = np.sum(diagonal_differences)
    
    
    sorted_rows = [sorted(row) for row in arr]  
    smallest_sorted_row = min(sorted_rows)  
    smallest_row_index = sorted_rows.index(smallest_sorted_row)  

    
    return result_arr, diagonal_sum, (smallest_row_index, smallest_sorted_row)



arr = np.array([
    [10, 15, 20, 25],
    [2, 18, 5, 12],
    [30, 25, 22, 10],
    [8, 14, 9, 7]
])


try:
    result_arr, diagonal_sum, (smallest_row_index, smallest_sorted_row) = analyze_complex_array(arr)
    
    print("Transformed Array:")
    print(result_arr)
    print("\nSum of Absolute Differences Between Diagonals:", diagonal_sum)
    print("\nSmallest Sorted Row (Index and Row):", smallest_row_index, smallest_sorted_row)
    
except ValueError as e:
    print(f"Error: {e}")

#%%
#16. Write a function `format_table(s)` which takes a string containing unstructured expense information, separated by comma,
#such as

#    ```
#    Dinner with Ted, 30.00, Bus back home,  2, Present for aunt, Mary 25.99
#    ```

#You should return a string which contains the same data as a formatted table, plus a final row for the total amount, 
#   which should print as shown:

#    ```
#    Dinner with Ted       30.00
#    Bus back home          2.00
#    Present for aunt Mary 25.99
#    TOTAL                 57.99
#    ```


def format_table(s):
    
    tokens = s.split(",")
    
    descriptions = [desc.strip() for desc in tokens[::2]]  
    prices = [float(p.strip()) for p in tokens[1::2]]  
    
    total = sum(prices)
    
    descriptions.append("TOTAL")
    prices.append(total)
    
    max_len_desc = max(len(desc) for desc in descriptions)
    max_len_price = max(len(f"{p:.2f}") for p in prices)
    
    result = ""
    for desc, price in zip(descriptions, prices):
        result += (
            desc
            + " " * (1 + max_len_desc - len(desc))  
            + f"{price:{max_len_price}.2f}"  
            + "\n"
        )
    
    return result.strip()

#%%
#17. Write a function `generate_quadratic_data(a, b, c, num_points, noise_std=0)` that: generates `num_points` 
#data points for the quadratic equation: `y = a * x**2 + b * x + c`, allows the addition of Gaussian noise
#with a standard deviation of `noise_std`, and returns x and y.
#In addition, write a function plot_quadratic_data(x, y, title=`Quadratic Data`) that: creates a scatter plot of
#the generated data.
#Saves the plot as a PDF file named quadratic.pdf.
#Use the following example values to generate the data and the plot:
#a = 1, b = -2, c = 1, num_points = 100, noise_std = 0.5
#To save the figure, call plt.savefig(`quadratic.pdf`) (make sure that you do this before or instead of plt.show(), 
#otherwise you will save a blank image).To upload the pdf to your github, open the git console and run `git add quadratic.pdf`
#before committing and pushing. This task will be graded manually.

import numpy as np
import matplotlib.pyplot as plt

def generate_quadratic_data(a, b, c, num_points, noise_std=0):
    x = np.linspace(-10, 10, num_points)
    y = a * x**2 + b * x + c
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, num_points)
        y += noise
    return x, y

def plot_quadratic_data(x, y, title='Quadratic Data'):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Generated Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('/the_path/quadratic.pdf')
    plt.show()

a, b, c = 1, -2, 1
num_points = 100
noise_std = 0.5  

x, y = generate_quadratic_data(a, b, c, num_points, noise_std)

plot_quadratic_data(x, y, title='Generated Quadratic Data')

#%%
#18. Write a function `find_root_of_poly(coeffs, a, b, tol)` that approximates a root of a degree $3$ polynomial contained 
#in the interval $[a,b]$ (the input polynomial your function will be tested on will indeed
#have a real root in the interval). The approximation is supposed to be within `tol` precision. The `coeffs` input will 
#be a list of the coefficients of the polynomial, starting with the coefficient of the free
#term and finishing with the coefficient of $x^3$.

import numpy as np

def find_root_of_poly(coeffs, a, b, tol):
    
    coeffs = coeffs[::-1]
    
    left = a
    right = b
    
    half = (left + right) / 2

    while (right - left) / 2 > tol:
        p_left = np.polyval(coeffs, left)
        p_half = np.polyval(coeffs, half)

        if p_left * p_half < 0:
            right = half
        else:
            left = half

        half = (left + right) / 2

    return half

#%%
#19. Write a function `get_average_distance_unit_square(n)` that estimates the average distance 
#between two random points (uniformly drawn) in the unit square based on $n$
#different random (uniform) draws of pairs of points in the unit square. 

import numpy as np
from numpy import random

def get_average_distance_unit_square(n):
    points1 = np.random.random((n, 2))  
    points2 = np.random.random((n, 2)) 

    distances = np.sqrt(np.sum((points2 - points1) ** 2, axis=1)) 

    average_distance = np.mean(distances)  
    return average_distance
#%%
#20. Write a function `get_average_distance_unit_disk(n)` that estimates the average distance 
#between two random points (uniformly drawn) in the unit disk based on $n$ different random (uniform) draws
#of pairs of points in the unit disk. If you intend to generate the random points via random polar coordinates,
#be very careful with how you use the random radial component when generating the points.
#Recall that the probability of a point landing in a particular subset of the disk should be proportional to that subset's area.

import numpy as np
from numpy import random

def get_average_distance_unit_disk(n):
    theta1 = random.uniform(0, 2 * np.pi, n)
    theta2 = random.uniform(0, 2 * np.pi, n)

    r1 = np.sqrt(random.uniform(0, 1, n))
    r2 = np.sqrt(random.uniform(0, 1, n))

    x1, y1 = r1 * np.cos(theta1), r1 * np.sin(theta1)
    x2, y2 = r2 * np.cos(theta2), r2 * np.sin(theta2)

    distances = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    average_distance = np.mean(distances)
    return average_distance
#%%
#21. Write a function `get_random_subset_of_naturals_up_to_20()` that outputs a 
#random subset of the set of integers between $1$ and $20$ in the format of a `numpy` array.
#The draw of the subset should be uniform, i.e., any subset should in principle have the same chance to be
#outputted by your function.
#This problem will be graded manually.
#For $2$ out of the $5$ points allotted to this problem, you can write your function however you wish. 
#To get $5$ points, your function is allowed to make a single call to the `numpy.random.randint()` method
#but it cannot make use of any other random methods.

import numpy as np
from numpy import random

def get_random_subset_of_naturals_up_to_20():
    subset_binary = np.random.randint(0, 2**20)
    subset = [i + 1 for i in range(20) if subset_binary & (1 << i)]
    return np.array(subset)
#%%
#22. Design a function math_pipeline that:
#Accepts two inputs:
#A list of operations `("add", "subtract", "multiply", "divide")` as strings.
#A single numeric operand that will be used for all operations.
#Returns a pipeline function that:
#Takes a starting value (start_value) as input.
#Applies the operations in the order specified in the list, using the operand each time. Returns the final result.
#error handling:
#if an invalid operation is provided, print an error message and return None.
#If a "divide" operation encounters a zero operand, print an error message and return None.
#If the list of operations is empty, the pipeline should act as an identity function, returning the starting 
#value unchanged.

def math_pipeline(operations, operand):
    def add(x):
        return x + operand  

    def subtract(x):
        return x - operand  

    def multiply(x):
        return x * operand  

    def divide(x):
        if operand == 0:  
            raise ZeroDivisionError("Cannot divide by zero")
        return x / operand  

    operation_map = {
        "add": add,
        "subtract": subtract,
        "multiply": multiply,
        "divide": divide,
    }

    for op in operations:
        if op not in operation_map:
            raise ValueError(f"Invalid operation: {op}")  
    def pipeline(start_value):
        result = start_value  
        for op in operations:
            result = operation_map[op](result)  
        return result  

    return pipeline  


operations = ["add", "multiply", "subtract", "divide"]
operand = 2  

pipeline = math_pipeline(operations, operand)

start_value = 10

result = pipeline(start_value)

print(f"Final result: {result}")
#%%
#%%
#23.Write two functions: `one_dimensional_random_walk(steps, up2down_chance_ratio)` 
#and `compare_random_walks(n, reference_pos, steps, up2down1, up2down2)`. The first should 
#simulate a simple one-dimensional random walk with `steps` many iterations where at each 
#iteration the chance to go up divided by the chance to go down (i.e. the probability of adding $1$ to 
#your previous position divided by the probability of subtracting $1$ from the previous position) 
#is the input `up2down_chance_ratio`. This function should return the ending position of the random walk. 

#The second function should compare how likely two different kinds of random walks are to stick close to a 
#desired ending position `reference_pos`. The second function should call the first $n$ times with parameters
#`steps` and `up2down1` and again $n$ times with parameters `steps` and `up2down2`. For each of
#the $n$ instances, the function is supposed to decide which of the $2$ random walks was closer to
#`reference_pos`. The second function should output two counters, representing how many times the 
#first type of random walk got closer to `reference_pos` and, respectively, how many times 
#the second type of random walk ended up closer. To get an intuition of what this means, 
#observe what happens when `steps` is $10000$, `reference_pos` is $100$, the first random walk 
#is uniform (i.e., the ratio `up2down1` is $1$) and the second random walk is slightly biased 
#towards going up (e.g. `up2down2` is $1.1$). Think about the expected value of the ending position 
#for each of these two types of random walks. 

#one_dimensional_random_walk: Simulates a random walk with a specified chance ratio.
#compare_random_walks: Compares two types of random walks in relation to a reference position.

import numpy as np 

def one_dimensional_random_walk(steps, up2down_chance_ratio):
    up_probability = up2down_chance_ratio / (1 + up2down_chance_ratio)
    
    random_steps = np.random.choice([-1, 1], size=steps, p=[1 - up_probability, up_probability])
    
    position = np.sum(random_steps)
    return position  

steps = 100
up2down_chance_ratio = 1.2  

final_position = one_dimensional_random_walk(steps, up2down_chance_ratio)
print(f"Final position after {steps} steps: {final_position}")
#%%
def compare_random_walks(n, reference_pos, steps, up2down1, up2down2):
    positions1 = np.array([one_dimensional_random_walk(steps, up2down1) for i in range(n)])
    
    positions2 = np.array([one_dimensional_random_walk(steps, up2down2) for i in range(n)])
    
    count_1 = np.sum(np.abs(positions1 - reference_pos) < np.abs(positions2 - reference_pos))
    
    count_2 = np.sum(np.abs(positions1 - reference_pos) > np.abs(positions2 - reference_pos))
    
    return (count_1, count_2)

## parameters
n = 1000  
reference_pos = 100  
steps = 10000  
up2down1 = 1  
up2down2 = 1.1  

## compare the two random walks
result = compare_random_walks(n, reference_pos, steps, up2down1, up2down2)

print(f"Walk 1 closer: {result[0]} times, Walk 2 closer: {result[1]} times")
#%%
#24. (From GTG R-2.5.)
#  The file `credit_card.py` contains the code of the credit card example from the book.
#  Revise the `charge` and `make_payment` methods to raise a `ValueError` if the caller does not send a number.


# Copyright 2013, Michael H. Goldwasser
#
# Developed for use with the book:
#
#    Data Structures and Algorithms in Python
#    Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser
#    John Wiley & Sons, 2013
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numbers

class CreditCard:
  """A consumer credit card."""

  def __init__(self, customer, bank, acnt, limit):
    """Create a new credit card instance.

    The initial balance is zero.

    customer  the name of the customer (e.g., 'John Bowman')
    bank      the name of the bank (e.g., 'California Savings')
    acnt      the acount identifier (e.g., '5391 0375 9387 5309')
    limit     credit limit (measured in dollars)
    """
    self._customer = customer
    self._bank = bank
    self._account = acnt
    self._limit = limit
    self._balance = 0

  def get_customer(self):
    """Return name of the customer."""
    return self._customer

  def get_bank(self):
    """Return the bank's name."""
    return self._bank

  def get_account(self):
    """Return the card identifying number (typically stored as a string)."""
    return self._account

  def get_limit(self):
    """Return current credit limit."""
    return self._limit

  def get_balance(self):
    """Return current balance."""
    return self._balance

  def charge(self, price):
    """Charge given price to the card, assuming sufficient credit limit.

    Return True if charge was processed; False if charge was denied.
    """
    if not isinstance(price, numbers.Number):
        raise ValueError(print(f"The passed object (price={price}) is not a number!"))
    if price + self._balance > self._limit:  # if charge would exceed limit,
      return False                           # cannot accept charge
    else:
      self._balance += price
      return True

  def make_payment(self, amount):
    """Process customer payment that reduces balance."""
    if not isinstance(amount, numbers.Number):
        raise ValueError(f"The passed object (amount={amount}) is not a number!")
    self._balance -= amount


if __name__ == '__main__':
  wallet = []
  wallet.append(CreditCard('John Bowman', 'California Savings',
                           '5391 0375 9387 5309', 2500) )
  wallet.append(CreditCard('John Bowman', 'California Federal',
                           '3485 0399 3395 1954', 3500) )
  wallet.append(CreditCard('John Bowman', 'California Finance',
                           '5391 0375 9387 5309', 5000) )

  for val in range(1, 17):
    wallet[0].charge(val)
    wallet[1].charge(2*val)
    wallet[2].charge(3*val)

  for c in range(3):
    print('Customer =', wallet[c].get_customer())
    print('Bank =', wallet[c].get_bank())
    print('Account =', wallet[c].get_account())
    print('Limit =', wallet[c].get_limit())
    print('Balance =', wallet[c].get_balance())
    while wallet[c].get_balance() > 100:
      wallet[c].make_payment(100)
      print('New balance =', wallet[c].get_balance())
    print()
#%%
#25. (From GTG R-2.9ff.)
#The file `vector.py` contains the code of the vector class from the book.
#Add the following methods to the class:
#`__sub__` so that the expression `u-v` returns a new vector instance
#representing the difference between two vectors
#and `__neg__` so that the expression `-v` returns a new vector instance
#whose coordinates are all the negated values of the respective coordinates of `v`.
#If the vectors do not have the same dimension, raise an error as for `__add__` in the example code.

#3. Continuing Problem 2:
#Implement the `__mul__` method so that `u*v` returns a scalar that represents the dot product of the vectors `u` and `v`,
#i.e., $u_1 \, v_1 + \cdots + u_d \, v_d$, and that `u*a` results in scalar multiplication if `a` is a number.
#Further, implement the `__rmul__` method to make sure that `a*v` is the same as `v*a` when `v` is a vector and `a` is a number.

#4. Continuing Problem 3:
#Implement the method `cross` so that `u.cross(v)` gives the cross product of the vectors `u` and `v`
#if both of their length is 3.
#Raise a value error otherwise.


# Copyright 2013, Michael H. Goldwasser
#
# Developed for use with the book:
#
#    Data Structures and Algorithms in Python
#    Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser
#    John Wiley & Sons, 2013
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import collections
import numbers

class Vector:
  """Represent a vector in a multidimensional space."""

  def __init__(self, d):
    if isinstance(d, int):
      self._coords = [0] * d 
    else:
      try:                                     
        self._coords = [val for val in d]
      except TypeError:
        raise TypeError('invalid parameter type')

  def __len__(self):
    """Return the dimension of the vector."""
    return len(self._coords)

  def __getitem__(self, j):
    """Return jth coordinate of vector."""
    return self._coords[j]

  def __setitem__(self, j, val):
    """Set jth coordinate of vector to given value."""
    self._coords[j] = val

  def __add__(self, other):
    """Return sum of two vectors."""
    if len(self) != len(other):          
      raise ValueError('dimensions must agree')
    result = Vector(len(self))           
    for j in range(len(self)):
      result[j] = self[j] + other[j]
    return result

  def __sub__(self, other):
    """Return the difference (a - b) of two vectors."""
    if len(self) != len(other):

        raise ValueError('dimensions must agree')
    res = Vector(len(self))
    res._coords = [x - y for (x,y) in zip(self._coords, other._coords)]
    return res

  def __mul__(self, other):
    """If `other` is a number:
    Return the vector `self` multiplied by `other` (in each coordinate).
    If `other` is of type `Vector`:
    Return the standard inner product of the two vectors."""
    if isinstance(other, numbers.Number):
        res = Vector(len(self))
        res._coords = [x * other for x in self._coords]
        return res
    if isinstance(other, Vector):
        if len(self) != len(other):
            
            raise ValueError('dimensions must agree')
        return sum((x * y for (x,y) in zip(self._coords, other._coords)))
    raise NotImplementedError("Multiplication not implemented for the given types!")

  def __rmul__(self, other):
    return self * other 

  def cross(self, other):
    """Return the cross product of the two given vectors."""
    if len(self) == 3 and len(other) == 3:
        res = Vector(3)
        a = self._coords
        b = other._coords
        res[0] = a[1] * b[2] - a[2] * b[1]
        res[1] = a[2] * b[0] - a[0] * b[2]
        res[2] = a[0] * b[1] - a[1] * b[0]
        return res
    raise ValueError("Cross product only defined for vectors of length 3.")

  def __neg__(self):
    """Return the negative (- b) of the given vector."""
    res = Vector(len(self))
    res._coords = [-x for x in self._coords]
    return res

  def __eq__(self, other):
    """Return True if vector has same coordinates as other."""
    return self._coords == other._coords

  def __ne__(self, other):
    """Return True if vector differs from other."""
    return not self == other             

  def __str__(self):
    """Produce string representation of vector."""
    return '<' + str(self._coords)[1:-1] + '>'  

  def __lt__(self, other):
    """Compare vectors based on lexicographical order."""
    if len(self) != len(other):
      raise ValueError('dimensions must agree')
    return self._coords < other._coords

  def __le__(self, other):
    """Compare vectors based on lexicographical order."""
    if len(self) != len(other):
      raise ValueError('dimensions must agree')
    return self._coords <= other._coords


def cross(u,v):
  return u.cross(v)


if __name__ == '__main__':
  # the following demonstrates usage of a few methods
  v = Vector(5)              # construct five-dimensional <0, 0, 0, 0, 0>
  v[1] = 23                  # <0, 23, 0, 0, 0> (based on use of __setitem__)
  v[-1] = 45                 # <0, 23, 0, 0, 45> (also via __setitem__)
  print(v[4])                # print 45 (via __getitem__)
  u = v + v                  # <0, 46, 0, 0, 90> (via __add__)
  print(u)                   # print <0, 46, 0, 0, 90>
  total = 0
  for entry in v:            # implicit iteration via __len__ and __getitem__
    total += entry   
#%%
#26. Use sympy to solve the system of equations $x^2+y^2=r^2$, $2y=4x+1$.
#Your code should define a list of dictionaries named `sol` which contains 
#replacement expressions for `x` and `y`
#in terms of the symbolic parameter `r`.

import sympy as sp

x, y, r = sp.symbols('x y r')

eq1 = sp.Eq(x**2 + y**2, r**2)
eq2 = sp.Eq(2 * y, 4 * x + 1)

sol = sp.solve([eq1, eq2], [x, y], dict=True)

sol
#%%
#27. Plot the two curves described by the equations from Problem 1 for $r=2$ into a single coordinate system.
#Further, plot the solution points derived in Problem 1 as two visible dots into the same coordinate axes.
#This question will be graded manually based on the graphical output.  
#Be sure to label your coordinate axes and choose a sensible coordinate range for plotting.  
#Before committing your submission, issue `git add Problem2.pdf` to make sure that your graph will be uploaded to Github.


from pylab import *

x, y, r = sp.symbols('x y r')


eq1 = sp.Eq(x**2 + y**2, r**2)  
eq2 = sp.Eq(2 * y, 4 * x + 1)   

solutions = sp.solve([eq1, eq2], [x, y], dict=True)


r_value = 2  
numerical_solutions = [{k: v.subs(r, r_value) for k, v in sol.items()} for sol in solutions]


solution_points = [(float(sol[x]), float(sol[y])) for sol in numerical_solutions]


theta = np.linspace(0, 2 * np.pi, 500)  
circle_x = r_value * np.cos(theta)  
circle_y = r_value * np.sin(theta)  


line_x = np.linspace(-2.5, 2.5, 500)  
line_y = (4 * line_x + 1) / 2         


plt.figure(figsize=(8, 8))


plt.plot(circle_x, circle_y, label='$x^2 + y^2 = r^2$', color='blue')  
plt.plot(line_x, line_y, label='$2y = 4x + 1$', color='green')         


for sx, sy in solution_points:
    plt.scatter(sx, sy, color='red', zorder=5, label=f'Solution ({sx:.2f}, {sy:.2f})')  


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.grid(True)
plt.legend(loc='upper left')  
plt.title('Plot of $x^2 + y^2 = r^2$ and $2y = 4x + 1$ for $r=2$')
plt.savefig('Problem2.pdf')
plt.show()
#%%
#28. Suppose you have a dataset representing the tasks completed by 100 employees
#in two different departments, A and B. Each department comprises 50 employees.
#The 'tasks_completed' column contains random integers between 10 and 25 
#(inclusive), indicating the number of tasks completed by each employee. 
#Based on the assumption, create a DataFrame and determine whether there is a 
#statistically significant difference in the average number of tasks completed 
#between Department A and Department B. Conduct an independent samples t-test 
#to compare the average tasks completed by employees in Department A and 
#Department B. Interpret the result considering a significance level (alpha) 
#of 0.05, and provide a box plot to visualize the distributions. 
#This question will be graded manually. Before committing your submission, 
#issue git add Problem1.pdf to make sure that your plot will be uploaded to Github.

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(0)
tasks_A = np.random.randint(10, 26, 50) 
tasks_B = np.random.randint(10, 26, 50)  

data = pd.DataFrame({
    'Department': ['A']*50 + ['B']*50,
    'Tasks_Completed': np.concatenate([tasks_A, tasks_B])
})

t_stat, p_value = stats.ttest_ind(data[data['Department'] == 'A']['Tasks_Completed'], 
                                  data[data['Department'] == 'B']['Tasks_Completed'])

alpha = 0.05
significant_difference = p_value < alpha

interpretation = "There is a statistically significant difference" if significant_difference \
                 else "There is no statistically significant difference"

plt.figure(figsize=(8, 6))
plt.boxplot([tasks_A, tasks_B], labels=['Department A', 'Department B'])
plt.title('Box Plot of Tasks Completed in Departments A and B')
plt.ylabel('Tasks Completed')
plt.grid(True)
plt.show()

t_stat, p_value, interpretation
#%%
#29. It is known that any rational function of $\sqrt{2}$ with rational coefficients 
#can be written in the canonical representation $r(\sqrt{2}) = a + b \sqrt{2}$, 
#where $a$ and $b$ are again rational numbers.

#Write a function `canonical_representation(r)` that takes as argument the 
#rational function $r$ as a sympy function, and returns the coefficients
#`(a,b)` as a Python tuple.

import sympy as sp

q = sp.sqrt(2)

def canonical_representation(r):
    s = sp.simplify(r(q))
    return s.subs(q,0), s.coeff(q)
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#https://docs.sympy.org/latest/tutorials/intro-tutorial/intro.html
#https://ethanweed.github.io/pythonbook/03.01-descriptives.html
