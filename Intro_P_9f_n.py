#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:55:26 2024

@author: maryamramezaniziarani
"""
#%%
class Person:
    def __init__(self, name): ## constructor method initializes attributes
        self.name = name

    def greet(self):
        return f"Hello, my name is {self.name}!"

## create an instance
person = Person("Alice")

## access attributes
print(f"name is {person.name}")

##(call method)
print(person.greet())

#%%
## Define a class named 'Dog'
class Dog:
    def __init__(self, name, age):
        ## constructor method initializes attributes
        self.name = name
        self.age = age

    def bark(self):
        ## method to make the dog bark, meant to perform an action related to the class
        print("Woof! Woof!")

## create instances (objects) of the 'Dog' class
dog1 = Dog("Buddy", 3)
dog2 = Dog("Charlie", 5)

## access attributes and call methods
print(f"{dog1.name} is {dog1.age} years old.")
dog1.bark()  

print(f"{dog2.name} is {dog2.age} years old.")
dog2.bark()  
#%%
#Exercise1:
    
#Create a Python class named BankAccount to model a simple bank account. 
#The class should have the following attributes and methods:

#Attributes:

#account_holder (string): the name of the account holder.
#balance (float): the current balance in the account.

#Methods:

#A constructor method to initialize the account with the account holder's name
#and an initial balance.
#A method to deposit a specified amount into the account.
#A method to withdraw a specified amount from the account.
#A method to retrieve the current balance of the account.

class BankAccount: # Constructor method initializes attributes
    def __init__(self, account_holder, initial_balance):
        self.account_holder = account_holder
        self.balance = initial_balance

    def deposit(self, amount):
        self.balance += amount
        print(f"Deposited ${amount:.2f}. Current Balance: ${self.balance:.2f}")

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            print(f"Withdrew ${amount:.2f}. Current Balance: ${self.balance:.2f}")
        else:
            print("Insufficient funds. Withdrawal canceled.")

    def get_balance(self):
        return self.balance

## Example  Create instances (objects) and call methods
my_account = BankAccount("Alice", 1000.0)
my_account.deposit(500.0)
my_account.withdraw(200.0)
my_account.deposit(100.0)

## Acess attributes and call method
final_balance = my_account.get_balance()
print(f"{my_account.account_holder}'s Final Balance: ${final_balance:.2f}")

#%%
#Inheritance
#%%
# superclass
class Vehicle:
    def __init__(self, brand, model): # Constructor method initializes attributes
        self.brand = brand
        self.model = model

    def start_engine(self):
        return "Engine started"

    def stop_engine(self):
        return "Engine stopped"


# Subclass1
class Car(Vehicle):
    def __init__(self, brand, model, num_doors):
        ## Call the constructor of the superclass using super()
        super().__init__(brand, model)
        self.num_doors = num_doors

    def start_engine(self):
        ## Override the start_engine method for Car
        return "Car engine started"


# Subclass2
class Bicycle(Vehicle):
    def __init__(self, brand, model, num_gears):
        ## Call the constructor of the superclass using super()
        super().__init__(brand, model)
        self.num_gears = num_gears

    def start_engine(self):
        ## Override the start_engine method for Bicycle
        return "Bicycle doesn't have an engine"


## create instances of the classes
car = Car(brand="Porsche", model="Panamera", num_doors=4)
bicycle = Bicycle(brand="Schwinn", model="Roadster", num_gears=7)

##access attributes and call methods
print(f"{car.brand} {car.model} with {car.num_doors} doors: {car.start_engine()}")
print(f"{bicycle.brand} {bicycle.model} with {bicycle.num_gears} gears: {bicycle.start_engine()}")
#%%
#Exercise2:
# Imagine you are tasked with designing a system to represent geometric shapes. Each shape has common
# properties (e.g., area, perimeter), but different shapes may have unique properties and behaviors. 
# Your goal is to create a class hierarchy to represent various geometric shapes.

# Requirements:

# Define a base class called Shape with the following methods:
# - area(): Returns the area of the shape.
# - perimeter(): Returns the perimeter of the shape.
#However, since each shape will calculate these differently, these methods should raise a NotImplementedError
#to ensure that subclasses provide their specific implementation.

# Implement three subclasses: Circle, Rectangle, and Triangle. Each subclass should inherit from the Shape class.

# For each subclass, implement the necessary methods to calculate the area and perimeter based on their 
#specific formulas:
# - Circle: area = πr^2, perimeter = 2πr
# - Rectangle: area = length × width, perimeter = 2(length + width)
# - Triangle: Use Heron's formula to calculate the area, perimeter = side1 + side2 + side3

# Create instances of each shape and demonstrate the use of their methods.

# Add error handling to ensure that the sides, radius, length, and width are non-negative values.

import math

# Base class
class Shape:
    def area(self):
        raise NotImplementedError("Subclasses must implement this method")

    def perimeter(self):
        raise NotImplementedError("Subclasses must implement this method")

# Subclasses
class Circle(Shape):
    def __init__(self, radius):
        if radius < 0:
            raise ValueError("Radius must be a non-negative value.")
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def perimeter(self):
        return 2 * math.pi * self.radius

class Rectangle(Shape):
    def __init__(self, length, width):
        if length < 0 or width < 0:
            raise ValueError("Length and width must be non-negative values.")
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * (self.length + self.width)

class Triangle(Shape):
    def __init__(self, side1, side2, side3):
        if side1 < 0 or side2 < 0 or side3 < 0:
            raise ValueError("All sides of the triangle must be non-negative values.")
        self.side1 = side1
        self.side2 = side2
        self.side3 = side3

    def area(self):
        s = (self.side1 + self.side2 + self.side3) / 2
        return math.sqrt(s * (s - self.side1) * (s - self.side2) * (s - self.side3))

    def perimeter(self):
        return self.side1 + self.side2 + self.side3

## Example
try:
    circle = Circle(radius=-5)
    rectangle = Rectangle(length=4, width=6)
    triangle = Triangle(side1=3, side2=4, side3=5)

    ## Demonstrate methods
    print(f"Circle Area: {circle.area()}, Perimeter: {circle.perimeter()}")
    print(f"Rectangle Area: {rectangle.area()}, Perimeter: {rectangle.perimeter()}")
    print(f"Triangle Area: {triangle.area()}, Perimeter: {triangle.perimeter()}")

except ValueError as e:
    print(f"Error: {e}")

#%%
#Python Magic Methods
#%%
class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def start_engine(self):
        return "Engine started"

    def stop_engine(self):
        return "Engine stopped"

    def __str__(self):
        return f"{self.brand} {self.model}"

    def __repr__(self):
        return f"Vehicle(brand={self.brand}, model={self.model})"


class Car(Vehicle):
    def __init__(self, brand, model, num_doors):
        super().__init__(brand, model)
        self.num_doors = num_doors

    def start_engine(self):
        return "Car engine started"

    def __str__(self):
        return f"{super().__str__()} with {self.num_doors} doors"

    def __repr__(self):
        return f"Car(brand={self.brand}, model={self.model}, num_doors={self.num_doors})"


class Bicycle(Vehicle):
    def __init__(self, brand, model, num_gears):
        super().__init__(brand, model)
        self.num_gears = num_gears

    def start_engine(self):
        return "Bicycle doesn't have an engine"

    def __str__(self):
        return f"{super().__str__()} with {self.num_gears} gears"

    def __repr__(self):
        return f"Bicycle(brand={self.brand}, model={self.model}, num_gears={self.num_gears})"


## create instances of the classes
car = Car(brand="Porsche", model="Panamera", num_doors=4)
bicycle = Bicycle(brand="Schwinn", model="Roadster", num_gears=7)

## call methods
print(str(car))
print(str(bicycle))

## For debugging purposes
print(repr(car))
print(repr(bicycle))

#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
