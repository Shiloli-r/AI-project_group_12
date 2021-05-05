import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = "Britam.csv"
data_set = pd.read_csv('test/Britam.csv')
print(data_set)

#  Describing the data
print(data_set.shape)
print(data_set.describe())

#  Exploratory data analysis #

# -------- Highest and Lowest price ------------
#data_set.plot(x="Low", y="High", style='o')
#plt.title("Lowest Price Vs High Price")
#plt.xlabel("Lowest Price")
#plt.ylabel("Highest Price")
#plt.show()

# Output variable
#plt.figure(figsize=(7, 4))
#plt.tight_layout()
#sb.distplot(data_set['High'])

# -------- Opening and Closing price ------------
data_set.plot(x="Open", y="Close", style='o')
plt.title("Open price Vs Close price")
plt.xlabel("Opening price")
plt.ylabel("Closing price")
plt.show()

# Output variable
plt.figure(figsize=(7, 4))
plt.tight_layout()
sb.distplot(data_set['Close'])
plt.show()

#  -------- Data Splicing -----------------------
x = data_set['Open'].values.reshape(-1, 1)
y = data_set['Close'].values.reshape(-1, 1)


# -------- Splitting the data into training and testing data -------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Training the algorithm
# importing the Regression class and instantiating it
regressor = LinearRegression()
regressor.fit(x_train, y_train)  # the fit method together with the training data trains the algorithm

# retrieving the y intercept
print("The y-intercept: ", regressor.intercept_)
print("The coefficient: ", regressor.coef_)  # for every unit change in the independent variable, the change in the
                                            # output variable is the coefficient

# Making Predictions on the data after training the model, using the test data
y_pred = regressor.predict(x_test)

df = pd.DataFrame({"Actual": y_test.flatten(), "predicted": y_pred.flatten()})
print(df)  # prints the data frame (to the console)

df1 = df.head(25)
df1.plot(kind="bar", figsize=(7, 5))
plt.grid(which='major', linestyle="-", linewidth="0.3", color="green")
plt.grid(which='minor', linestyle=":", linewidth="0.3", color="black")
plt.show()

plt.scatter(x_test, y_test, color="gray")
plt.plot(x_test, y_pred, color="red", linewidth=2)
plt.show()

# Evaluating the performance of the algorithm
print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: ", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))