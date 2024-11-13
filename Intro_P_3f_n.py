#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 2024

@author: maryamramezaniziarani
"""

##Lecture 1 and 2 Recap

##Variables and data types
#%%
# Numeric int, float
#%%
a=1
type(a)
#%%
b= 2.8
type(2.8)
(f"float type: {type(b)}")
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
##Excercise1: Create one variable for each basic data type and identify 
#its type using f" for example; 
#print(f"Integer type: {type(integer_n)}") Integer type: <class 'int'>

# Numeric
integer_n = 42
float_n = 3.14

# String 
text = "Hello"
# List 
my_list = [1, 2, 3, 4, 5]
# Tuple
my_tuple = (1, "apple", 3.14)
# Dictionary 
my_dict = {"name": "John", "age": 25}
# Set
my_set = {1, 2, 3, 4, 5}
# Boolean
is_test = True

# Check all types
print(f"Integer type: {type(integer_n)}")
print(f"Float type: {type(float_n)}")
print(f"String type: {type(text)}")
print(f"List type: {type(my_list)}")
print(f"Tuple type: {type(my_tuple)}")
print(f"Dictionary type: {type(my_dict)}")
print(f"Set type: {type(my_set)}")
print(f"Boolean type: {type(is_test)}")

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
my_string[0] = 'h'  ## This will raise an error
#%%
##Excercise2: Creates a list of numbers and add a new element to your list,
#use .append() method.

## Original list
numbers = [1, 2, 3, 4, 5]
print("Original list:", numbers)

# Add a number
numbers.append(6)
print("After adding 6:", numbers)
#%%
#Excercise3: Check this for (dict, set, tuple)

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
# Conditional Statements (if-elif-else)

grade=60

if grade>50:
   print("you pass the test")
#%%
a=34
if a<20:
    print("you are a winner")
elif a<60:
    print("you are a winner")
else:
    print("you are a loser")

#%%
#Excercise4: Write a code that classifies a given number as positive, negative, 
#or zero. (print the result using f")

num = 7  

# Check positive/negative/zero
if num > 0:
    sign = "positive"
elif num < 0:
    sign = "negative"
else:
    sign = "zero"

print(f"The number {num} is {sign}.")

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
#Exercise5: Write a Python code to find all the numbers in a given list that are greater 
#than a specified value. print the result with f"


numbers = [10, 25, 30, 15, 5, 50]
specified_value = 20


greater_numbers = []

for num in numbers:
    if num > specified_value:
        greater_numbers.append(num)

# Print the result
print(f"Numbers greater than {specified_value}: {greater_numbers}")

#%%
#while loop
#%%
i = 1
while i < 11:
    print(i)
    i += 1
#%%
#Exercise6: Write a Python program that repeatedly asks the user to enter a number until 
#they enter a negative number. 
#Once a negative number is entered, the program should print the sum of all 
#positive numbers entered

total_sum = 0
number = 0

## keep asking for numbers until a negative number is entered
while number >= 0:
    number = int(input("Enter a number (enter a negative number to stop): "))
    
    if number >= 0:
        total_sum += number

print("Sum of positive numbers:", total_sum)

#%%
## Def function
def add_numbers(num1, num2):
    sum = num1 + num2
    return sum
print(add_numbers(3, 5))

#%%
#Exercise7: Write a Python function that takes a sentence as input and returns a
#dictionary
#where the keys are the words in the sentence, and the values are the lengths of 
#those words.
#use .split() method to split sentence into words. (.split is used to divide a 
#string into a 
#list of substrings based on a specified delimiter (by default, whitespace)). 
#also, len() 
#function can be used to determine the length

#For example:

#Input: "I love python"
#Output: {'I': 1, 'love': 4, 'python': 6}

def word_lengths(sentence):
    words = sentence.split()  
    lengths = {}  
    for word in words:
        lengths[word] = len(word)  
    return lengths

print (word_lengths("I love python"))
#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
