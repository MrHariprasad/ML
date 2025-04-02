# Importing Necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate random data-set
np.random.seed(0)
x = np.random.rand(100, 1)  # Generate a 2-D array with 100 rows, each row containing 1 random number
y = 2 + 3 * x + np.random.rand(100, 1)

# Model initialization
regression_model = LinearRegression()

# Fit the data (train the model)
regression_model.fit(x, y)

# Predict
y_predicted = regression_model.predict(x)

# Model evaluation
rmse = np.sqrt(mean_squared_error(y, y_predicted))  # RMSE is the square root of MSE
r2 = r2_score(y, y_predicted)

# Printing values
print('Slope:', regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error:', rmse)
print('R2 score:', r2)

# Plotting values
# Data points
plt.scatter(x, y, s=10)
plt.xlabel('x-Values from 0-1')
plt.ylabel('y-values from 2-5')

# Predicted values
plt.plot(x, y_predicted, color='r')

plt.show()



#output
Slope: [[2.93655106]]
Intercept: [2.55808002]
Root mean squared error: 0.2761036867351649
R2 score: 0.9038655568672764
