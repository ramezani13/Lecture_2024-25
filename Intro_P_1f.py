#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 16 15:17:29 2024

@author: Maryam Ramezani Ziarani
"""

##Variables and data types
#%%
# Numeric int, float
#%%
a=1
type(a)
#%%
b= 2.8
type(2.8)
#%%
# Sequence list, tuple
#%%
type([4, 6, 7])
#%%
type((1, 5))
#%%
# String str
#%%
type("I like Math")
#%%
# Mapping dict
#%%
type({"number of student": 8, "p": 5})
#%%
#set
#%%
type({"number of student", "p"})
#%%
#Boolean 
#%%
print(type(1 == 2))
#%%
c=("I like Math!")
print(c[2])
#%%
#Excercise1: Assign a list to a variable and check the type and 
#print an element using index!
#%%
b=[4,5,8]
type(b)
print(b[2])
#%%
#Excercise2: Assign a float to a variable and use the print() function
#to print out the following sentence "the type of 2.8 is <class 'float'>"
#%%
b= 2.8
print ("the type of", b ,"is", type(b))
#%%
# Identity and Equality
#%%

a = [1, 2, 3]
b = a               ## b is a reference to the same list as a
c = [1, 2, 3]       ## c is a different list with the same values as a

identity_check = print(a is b)    ## true, they reference the same object
equality_check = print(a == c)    ## true, they have the same values

#%%
# Mutable (list, dict, set) vs. Immutable (string, numbers, tuple)
#%%
## Lists are mutable
my_list = [1, 2, 3]
my_list[0] = 4
print(my_list)

## Strings are immutable
my_string = "Hello"
# my_string[0] = 'h'  ## This will raise an error


#%%
#Excercise3: Check this for (dict, set, tuple)
#%%
## Dictionaries are mutable
my_dict = {'name': 'Alice', 'age': 25}
my_dict['age'] = 26
print(my_dict)  

## Sets are mutable
my_set = {1, 2, 3}
my_set.add(4)
print(my_set)  

## Tuples are immutable
my_tuple = (1, 2, 3)
my_tuple[0] = 4  
print(my_tuple)

#%%    
# Arithmetic operators
#%%
a = 5
b = 2
addition = a + b
print(a+b)
subtraction = a - b
print(a-b)
multiplication = a * b
print(a*b)
division = a / b
print(a/b)
remainder = a % b
print(a%b)
exponentiation = a ** b
print(a**b)

#%%
# Comparison operators
#%%
x = 5
y = 10
equals = x == y
not_equals = x != y
greater_than = x > y
less_than = x < y
print(x > y)
#%%
#Conditional Statements
#%%
grade=60

if grade>50:
   print("you pass the test")
#%%
a=34
if a<20:
    print("you are a winner")
else:
    print("you are a loser")
#%% #IndentationError
grade=60

if grade>50:
    print("you pass the test")
#%%
#Excercise4: Given two numbers a and b, write a Python code
#to check if a is divisible by b or not.
#%%
a=10
b=5
if (a % b == 0): 
    print("a is divisible by b") 
else: 
    print("a is not divisible by b")    
#%%
# for and while loops
#%%
animals = ["dog", "cat", "bird"]
for x in animals:
    print(x)
#%%
animals = ["dog", "cat", "bird"]
for x in animals:
    print(x)
    if x == "cat":
        break
#%%
x = range(10, 20)
for n in x:
    print(n)
#%%
x = range(10, 20,2)
for n in x:
    print(n)
#%%
#Exercise5: What numbers between 2 and 30 are even and odd?
#%%
for x in range(2, 30):  
    if x % 2 == 0: 
        print(x, "is a Even Number")
    else: 
        print(x, "is a Odd Number")
#%%
#Exercise6: Write a Python program to find the largest number in a 
#given list of numbers
#%%
arr = [2, 4, 6, 8]
max_num = arr[0]

for num in arr:
    if num > max_num:
        max_num = num
print(max_num)
#%%
#while loop
#%%
i = 1
while i < 11:
    print(i)
    i += 1
#%%
#Exercise7: print out the numbers from 100 to 0 in steps of 10, 
#using while loop
#%%
i = 100
while i >= 0:
    print(i)
    i -= 10
#%%
## Def function
def add_numbers(num1, num2):
    sum = num1 + num2
    return sum
print(add_numbers(3, 5))
#%%
#Exercise8: What is the area of a rectangle with a length of 2 and a 
#width of 4? (use def function)
#%%
def area_of_rectangle(length, width):
    area = length * width
    return area
print(area_of_rectangle(2, 4))
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org