#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:44:44 2025

@author: maryamramezaniziarani

"""
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
#%% optimize

##finding the minimum of a single variable function

def f1(x):
    return x**2 + 4*x + 4

## optimiz the function
result1 = optimize.minimize_scalar(f1)
print("Minimum of x^2 + 4x + 4 is at x =", result1.x)
#%% multivariable optimization 

def f2(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

## optimiz the function 
result2 = optimize.minimize(f2, [0, 0])

print("Minimum point for (x-1)^2 + (y-2)^2:", result2.x)

##This minimum occurs at the point x and y

#%%
### #Exercise1: The profit function P(x) for a company producing a single product is given by:

#P(x) = -2x^2 + 16x - 30

#Where x is the number of units produced.

#You are required to:
    
#1. Write a Python function for P(x).
#2. Use scipy.optimize.minimize_scalar to find the value of x that maximizes the profit. 
#(Hint: Since minimize_scalar minimizes, modify the function to minimize the negative profit instead.)
#3. Plot the profit function using matplotlib, marking the maximum point clearly.
#4. Use an if statement to check if the optimal x-value is within the range 0 ≤ x ≤ 10.
#5. If x is outside the range, print a message indicating it is invalid.



## define the profit function
def profit(x):
    return -2 * x**2 + 16 * x - 30

## modify the function to minimize the negative profit
def negative_profit(x):
    return -profit(x)

## use minimize_scalar to find the value of x that maximizes the profit
result = optimize.minimize_scalar(negative_profit)

## extract the optimal x and calculate the maximum profit
optimal_x = result.x
max_profit = profit(optimal_x)

## check if the optimal x is within the feasible production range
if 0 <= optimal_x <= 10:
    print(f"The optimal number of units to produce is x = {optimal_x:.2f}, which gives a maximum profit of ${max_profit:.2f}.")
else:
    print(f"The optimal value x = {optimal_x:.2f} is outside the feasible range (0 ≤ x ≤ 10).")

## visualize the profit function
x_values = np.linspace(0, 10, 500)  ## generate x values from 0 to 10
y_values = profit(x_values)         ## calculate profit for each x value

plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label="Profit Function", color="blue")
plt.scatter(optimal_x, max_profit, color="red", label=f"Maximum Profit: x={optimal_x:.2f}, P=${max_profit:.2f}")
plt.title("Profit Function: P(x) = -2x^2 + 16x - 30")
plt.xlabel("Number of Units Produced (x)")
plt.ylabel("Profit (P)")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.legend()
plt.grid(alpha=0.4)
plt.show()
#%%#statistic
from scipy import stats

data = [1, 2, 2, 3, 4, 4, 4, 5, 6]

## Compute mean, median, and mode
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode})")

#%%# Hypothesis Testing

##One-sample t-test

## Test if the sample mean is significantly different from a population mean
sample = [2.5, 3.0, 2.8, 3.5, 2.9]
population_mean = 3.0

t_stat, p_value = stats.ttest_1samp(sample, population_mean)
print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")

if p_value < 0.05:
    print("Reject the null hypothesis (sample mean is significantly different).")
else:
    print("Fail to reject the null hypothesis.")
#%%
#two-sample t-test

## Compare the means of two samples
sample1 = [2.1, 2.5, 3.0, 2.8, 3.2]
sample2 = [3.5, 3.8, 3.0, 3.7, 3.6]

t_stat, p_value = stats.ttest_ind(sample1, sample2)
print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")

#%%
### #Exercise2: You are tasked with analyzing the daily steps data of a group of 200 participants.
#The dataset is generated randomly using a uniform distribution, where the number of steps 
#is equally likely to fall anywhere between 5000 and 10000.

##Tasks:

#1. **Data generation**:
#- Generate a dataset of 200 daily step counts using a uniform distribution between 5000 and 10000
#steps.
#- Ensure the data is in integer format.

#2. **Preprocessing**:
#- Identify the **top 10% most active participants** (steps greater than the 90th percentile).

#3. **Hypothesis Testing**:
#- Test whether the average daily steps are significantly diferent from **7500 steps/day**:
#- Use a **one-sample t-test** to test the null hypothesis H0: "The mean daily steps are 7500/day."
#- Print the t-statistic, p-value, and conclusion:
#- Reject H0 if p < 0.05, otherwise fail to reject H0.


## Step 1: Data Generation
num_participants = 200
#steps = np.random.uniform(low=5000, high=10000, size=num_participants).astype(int)
steps = np.random.randint(low=5000, high=10000, size=200)
## Step 2: Preprocessing
percentile_90 = np.percentile(steps, 90)
most_active = steps[steps > percentile_90]

print(f"Generated Steps Data (Uniform Distribution): {steps}")
print(f"Top 10% most active participants (Steps > {percentile_90}): {most_active}")

## Step 3: Hypothesis Testing
t_stat, p_value = stats.ttest_1samp(steps, popmean=7500)

print(f"\nT-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < 0.05:
    print("Reject the null hypothesis: The mean daily steps are significantly different from 7500.")
else:
    print("Fail to reject the nul hypothesis: The mean daily steps are not significantly different from 7500.")

#%%# Correlation and Regression

# Two datasets
x = [10, 20, 30, 40, 50]
y = [15, 25, 35, 45, 55]

## compute Pearson correlation coefficient
corr, p_value = stats.pearsonr(x, y)
print(f"Pearson correlation coefficient: {corr:.2f} and p_value {p_value:.2f}")

#%%
##Linear regression models and predicts the relationship between a dependent and independent variable.
# Data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

## perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f"Slope: {slope:.2f}, Intercept: {intercept:.2f}")
print(f"R-squared: {r_value**2:.2f}")

## plot the regression line
plt.scatter(x, y, label="Data")
plt.plot(x, slope*x + intercept, color='red', label="Fitted Line")
plt.legend()
plt.show()
print(f"Regression Equation: y = {slope:.2f}x + {intercept:.2f}")
#%%
from scipy.optimize import curve_fit

## define an exponential function
def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c

## Example data (exponential relationship)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([1, 2.7, 7.4, 20.1, 54.6, 148.4])

## Fit the data
params, _ = curve_fit(exponential_model, x, y)
print(f"Fitted parameters: {params}")

## generate predictions
y_pred = exponential_model(x, *params)


plt.scatter(x, y, label="Data")
plt.plot(x, y_pred, color="red", label="Exponential Fit")
plt.legend()
plt.show()

rmse = np.sqrt(np.mean((y - y_pred) ** 2))
print(f"RMSE: {rmse:.2f}")
#%% 

### #Exercise3: Quadratic Curve Fitting

#### Dataset:
#x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
#y = [2, 5, 10, 17, 26, 37, 50, 65, 82, 101]

#### Tasks:
#1. Define the quadratic model: ( y = a * x**2 + b * x + c).
#2. Fit the quadratic model using `scipy.optimize.curve_fit`:
#   - Print the parameters a, b, and c .
#   - Plot the original data and the fitted curve.
#3. Fit the quadratic model using `numpy.polyfit`:
#   - Print the parameters a, b, and c .
#   - Plot the original data and the fitted curve.
#4. Compare the fitted parameters and curves from both methods.

## define the quadratic model
def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c

## dataset
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 5, 10, 17, 26, 37, 50, 65, 82, 101])

## fit the quadratic model
params, _ = curve_fit(quadratic_model, x, y)
print(f"Fitted parameters: a = {params[0]:.2f}, b = {params[1]:.2f}, c = {params[2]:.2f}")

## generate predictions
y_pred = quadratic_model(x, *params)


plt.scatter(x, y, label="Data")
plt.plot(x, y_pred, color="red", label="Quadratic Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Quadratic Curve Fitting")
plt.show()

#%%
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2, 5, 10, 17, 26, 37, 50, 65, 82, 101])

## fit a quadratic model using numpy.polyfit
coefficients = np.polyfit(x, y, deg=2)  # deg=2 for quadratic fit
a, b, c = coefficients
print(f"Fitted parameters: a = {a:.2f}, b = {b:.2f}, c = {c:.2f}")

## generate predictions using the fitted model
y_pred = np.polyval(coefficients, x)


plt.scatter(x, y, label="Data")
plt.plot(x, y_pred, color="red", label="Quadratic Fit (polyfit)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Quadratic Curve Fitting Using numpy.polyfit")
plt.show()
#%%
### small project: Sales Analysis for an E-Commerce Platform

#### Scenario
#You are working as a data analyst for an e-commerce platform. The company want to analyze
#its sales data to identify trends, relationships,and predict future performance.

##The dataset contains the following information:
#- `items_sold`: A list of the number of items sold each day.
#- `ad_spend`: A list of the daily advertising spend in dollars.

#You are tasked with analyzing this data to:
#1. Compute key statistics.
#2. Identify relationships between advertising spend and sales.
#3. Fit models for predictions.

### Tasks

#### **1. Preprocessing**
#1. Create a dictionary where the keys are the days (e.g., "Day 1", "Day 2") and the values 
#are tuples containing `(items_sold, ad_spend)`.
#2. Use list comprehensions to:
#- Extract all days where more than 100 items were sold.
#- Calculate the total sales revenue assuming each item is sold for $20.


#### **2. Correlation Analysis**
#1. Compute the Pearson correlation coefficient between `ad_spend` and `items_sold`.
#2. Interpret the result:
#- Is there a strong or weak relationship between advertising and sales?


#### **3. Linear Regression**
#1. Fit a linear regression model to predict `items_sold` based on `ad_spend`.
#2. Print:
#- The slope and intercept of the regression line.
#- The R^2  value to assess the model's fit.
#3. Predict how many items will be sold if $300 is spent on advertising.


#### **4. Curve Fitting**
#1. Assume sales follow an exponential growth model:
#\[ y = a \cdot e^{b \cdot x} + c\] 
#2. Use `curve_fit` to fit the exponential model to the data.
#3. Print the optimized parameters a, b , and  c .
#4. Predict the number of items sold if $300 is spent on advertising.



#### **5. Visualization**
#1. Plot the data as a scatter plot.
#2. Overlay both the linear regression line and the exponential curve.
#3. include proper labels and a legend.



### Dataset

#items_sold = [50, 60, 70, 90, 120, 150, 200, 250, 300, 350]
#ad_spend = [50, 60, 70, 80, 100, 120, 150, 180, 210, 250]

#%%References
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#R. Johansson, Numerical Python: Scientific Computing and Data Science Applications with Numpy, SciPy and Matplotlib, Apress, 2019 (“Joh”)
#J.V. Guttag, Introduction to Computation and Programming Using Python, third edition, The MIT Press, 2021
#https://docs.python.org
#For the assignments associated with this lecture (82-105-DS02, Introduction to Programming);
#M.T. Goodrich, R. Tamassia, and M.H. Goldwasser, Data Structures and Algorithms in Python, Wiley, 2013 (“GTG”)
#https://docs.sympy.org/latest/tutorials/intro-tutorial/intro.html
#https://ethanweed.github.io/pythonbook/03.01-descriptives.html
