import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tkinter import ttk
from tkinter import *

root = Tk()
root.title(" Predicting Nairobi Stock prizes")
root.geometry("1300x1000")

scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)
# scrollbar.config()
main = Frame(width=768, height=1200, bg="", background="black", colormap="new")
main.pack(fill=X, padx=5, pady=5)


title_label = Label(main, text="Predicting Nairobi Stocks Data", font="Helvetica 18 bold", background="black",
                    fg="white").place(x=500, y=5)
choose_title = Label(root, text="Choose Data Set ", font="Helvetica 10 bold", fg="green", background="black").place(
    x=300, y=50)
companies = ["Britam", "Safaricom", "Equity Group", "KCB", "Centum", "EABL", "Barclays Bank", "KQ",
                               "KenyaRE", "Kenya Power", "Cooperative Bank"]
options = ttk.Combobox(main,
                       values=companies)
options.place(x=500, y=50)
options.current(0)

explore_label = Label(main, text="Exploratory Analysis of Selected Stocks Data", font="Calibri 16 bold",
                      background="black", fg="red").place(x=450, y=80)
actual_label = Label(main, text="Actual Stock Data", font="Calibri 12 bold", background="black", fg="orange").place(
    x=250, y=100)

display = Text(main, width=77, height=15, relief=SUNKEN)
display.place(x=10, y=140)

description_label = Label(main, text="Data Description", font="Calibri 12 bold", background="black", fg="orange").place(
    x=950, y=100)
describe = Text(main, width=77, height=15, relief=SUNKEN)
describe.place(x=645, y=140)


# ------------------- Reading the data set --------------------------


def read_set():
    data = "database/" + options.get() + ".csv"
    data_set = pd.read_csv(data)
    return data_set


def slct():
    display.config(state="normal")
    describe.config(state="normal")
    display.delete('1.0', 'end')
    describe.delete('1.0', 'end')
    data = read_set()
    display.insert(INSERT, data)
    display.insert(END, "\n")
    describe.insert(INSERT, data.describe())
    describe.insert(END, "\n")
    display.config(state="disabled")
    describe.config(state="disabled")


select = Button(main, text="Select", width=10, fg="green", command=slct).place(x=750, y=50)

select_title = Label(root, text="Select Variables ", font="Helvetica 10 bold", fg="green", background="black").place(
    x=380, y=400)
choices = ["Opening vs Closing Price", " Lowest vs Highest price"]
variables = ttk.Combobox(main, values=choices, width=30)
variables.place(x=500, y=400)
variables.current(0)


def read_var():
    fetch = variables.get()
    if fetch == choices[0]:
        var = 1
    else:
        var = 2
    return var


select_var = Button(main, text="Select", width=10, fg="green", command=read_var).place(x=750, y=400)


def raw_plot():
    data = read_set()
    # -----------------------------------------------------
    if read_var() == 1:
        output_variable = "Closing Price"
        out = "Close"
        input_variable = "Opening Price"
        inp = "Open"
        title = "Opening Price vs Closing Price"
    else:
        output_variable = "Highest Price"
        out = "High"
        input_variable = "Lowest Price"
        inp = "Low"
        title = "Lowest Price vs Highest Price"
    # -------------------------------------------------------
    data.plot(x=inp, y=out, style='o')
    plt.title(title)
    plt.xlabel(input_variable)
    plt.ylabel(output_variable)
    plt.show()


def out_plot():
    data = read_set()
    # -------------------------------------------------------
    if read_var() == 1:
        out = "Close"
    else:
        out = "High"
    # --------------------------------------
    plt.figure(figsize=(7, 4))
    plt.tight_layout()
    sb.distplot(data[out])
    plt.show()


raw_data = Button(main, text="View Graphical Distribution of data set", fg="green", command=raw_plot).place(x=400,
                                                                                                            y=430)
out_data = Button(main, text="View Distribution of Output variable data", fg="green", command=out_plot).place(x=700,
                                                                                                              y=430)


# ------------------------------ DATA SPLICING AND TRAINING -------------------------------------------
def train():
    data = read_set()

    # --------------------------------------------------------
    if read_var() == 1:
        out = "Close"
        inp = "Open"

    else:
        out = "High"
        inp = "Low"

    # --------------------------------------------------------
    #  -------- Data Splicing -----------------------
    x = data[inp].values.reshape(-1, 1)
    y = data[out].values.reshape(-1, 1)

    # -------- Splitting the data into training and testing data -------------------------
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Training the algorithm
    # importing the Regression class and instantiating it
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)  # the fit method together with the training data trains the algorithm

    # retrieving the y intercept
    predictions.config(state="normal")
    predictions.delete('1.0', 'end')
    predictions.insert(INSERT, "The y-intercept: ")
    predictions.insert(INSERT, regressor.intercept_)
    predictions.insert(END, "\n")
    predictions.insert(INSERT, "The coefficient: ")
    predictions.insert(INSERT, regressor.coef_)  # for every unit change in the independent variable, the change in the
    predictions.insert(END, "\n")

    # output variable is the coefficient

    # Making Predictions on the data after training the model, using the test data
    y_pred = regressor.predict(x_test)

    df = pd.DataFrame({"Actual": y_test.flatten(), "predicted": y_pred.flatten()})
    predictions.insert(INSERT, df)
    predictions.config(state="disabled")
    return df


def pred():
    df = train()
    df1 = df.head(25)
    df1.plot(kind="bar", figsize=(7, 5))
    plt.grid(which='major', linestyle="-", linewidth="0.3", color="green")
    plt.grid(which='minor', linestyle=":", linewidth="0.3", color="black")
    plt.show()


def linefit():
    # ------------------ Redundant code ---------------------
    data = read_set()

    if read_var() == 1:
        out = "Close"
        inp = "Open"

    else:
        out = "High"
        inp = "Low"

    #  -------- Data Splicing -----------------------
    x = data[inp].values.reshape(-1, 1)
    y = data[out].values.reshape(-1, 1)

    # -------- Splitting the data into training and testing data -------------------------
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Training the algorithm
    # importing the Regression class and instantiating it
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)  # the fit method together with the training data trains the algorithm

    # Making Predictions on the data after training the model, using the test data
    y_pred = regressor.predict(x_test)

    # -------------------------------------------------------
    plt.scatter(x_test, y_test, color="gray")
    plt.plot(x_test, y_pred, color="red", linewidth=2)
    plt.show()


def evaluation():
    # ----------------- Redundant Code ------------
    data = read_set()
    if read_var() == 1:
        out = "Close"
        inp = "Open"

    else:
        out = "High"
        inp = "Low"

    #  -------- Data Splicing -----------------------
    x = data[inp].values.reshape(-1, 1)
    y = data[out].values.reshape(-1, 1)

    # -------- Splitting the data into training and testing data -------------------------
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Training the algorithm
    # importing the Regression class and instantiating it
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)  # the fit method together with the training data trains the algorithm

    # Making Predictions on the data after training the model, using the test data
    y_pred = regressor.predict(x_test)

    # -------------------------------------------------------

    performance.config(state="normal")
    performance.delete('1.0', 'end')
    performance.insert(INSERT, "Mean Absolute Error: ")
    performance.insert(INSERT, metrics.mean_absolute_error(y_test, y_pred))
    performance.insert(END, "\n")
    performance.insert(INSERT, "Mean Squared Error: ")
    performance.insert(INSERT, metrics.mean_squared_error(y_test, y_pred))
    performance.insert(END, "\n")
    performance.insert(INSERT, "Root Mean Squared Error: ")
    performance.insert(INSERT, np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    performance.config(state="disabled")


train_label = Label(main, text="Training and Prediction  Algorithm", font="Calibri 16 bold", background="black",
                    fg="red").place(x=100, y=460)

train_b = Button(main, text="Train and Test Data", fg="green", command=train).place(x=50, y=500)
predict = Button(main, text="Prediction Graph", fg="green", command=pred).place(x=50, y=530)
line = Button(main, text="Line of Best Fit", fg="green", command=linefit).place(x=50, y=560)
predictions = Text(main, width=45, height=15, relief=SUNKEN)
predictions.place(x=180, y=500)

evaluate_label = Label(main, text="Evaluating  Algorithm Performance", font="Calibri 16 bold", background="black",
                       fg="red").place(x=750, y=460)
evaluate = Button(main, text="Evaluate", fg="green", command=evaluation).place(x=600, y=500)
performance = Text(main, width=45, height=15, relief=SUNKEN)
performance.place(x=680, y=500)

root.mainloop()
