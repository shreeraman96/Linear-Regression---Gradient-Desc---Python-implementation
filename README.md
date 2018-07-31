# Linear-Regression---Gradient-Desc---Python-implementation
Implementing the mathematical model of linear regression with single feature in python

### The program is an attempt to implement the mathematical model of linear regression with single variable in python

### Files included :
	* ex1data1.txt - Dataset 
	* Linear Regression algorithm implementation in python.ipynb - Python script in jupyter notebook with the required outputs and interpretations
	* Linear Regression algorithm implementation in python.py - Python script that provides the required output
	
#### The Problem statement was obtained from Machine learning by Andrew NG course

#### Problem - Predicting the profit given the population

#### Problem type - Linear Regression with Single variable

#### Machine learning algorithm used : Linear Regression with Gradient Descent

#### Language : Python

#### Libraries used :
	* Pandas
	* Numpy
	* matplotlib

### Algorithm :
	* Import the required libraries
	* Read the dataset as data frame and then split the data into feature vector and output vector.
	* Visualize the data using a scatter plot.
	* Vectorize the feature vector by adding vector of one's.
	* Initialize the theta vector,learning rate and number of iterations.
	* Define function "computecost" that inputs feature vector,output vector and theta and computes the cost accordingly.
	* Define function "gradientdesc" that inputs feature vector,output vector,theta,learning rate and number of iterations and performs gradient desc and returns Costfunction history and converged theta.
	* Plot the cost function (J_hist) to check if the gradient descent has converged correctly.
	* Print the final Theta values and visualize the hypothesis function.
	* predict the profit for the desired population - 35000 and 70000.
	* visualize the cost function.

#### Variables - codebook:
	* x - population vector
	* y - profit vector
	* X - final vectorized feature vector for calculations
	* theta - parameter vector
	* alpha - learning rate
	* iteration - Number of iterations
	* J - value of cost function
#### Functions defined :

##### computecost 
	###### Input - X,y,theta
	###### Output - J

##### gradientdesc
	###### Input - X,y,theta,alpha,iterations
	###### Output - theta,J_hist
	