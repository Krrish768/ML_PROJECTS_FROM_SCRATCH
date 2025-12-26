## IMPORTING IMPORTANT LIBRARIES
import pandas as pd
import numpy as np
import random as r
from sklearn.model_selection import train_test_split

## DATA CLEANING AND PREPROCESSING
df=pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\ML_PROJECTS_FROM_SCRATCH\INSURANCE_PROJECT\insurance.csv")
print("Raw Data\n")
print(df)
print("\n\n")
df.sex=df["sex"].astype(str).str.strip().str.lower().map({"male":1,"female":0})
df.smoker=df["smoker"].astype(str).str.strip().str.lower().map({"yes":1,"no":0})
print("Cleaned Data\n")
print(df)
print("\n\n")
df.info()
print("\n")
print(df.describe())
print("\n")

x=df[["age", "sex", "bmi", "children", "smoker"]].to_numpy()
y=df[["charges"]].to_numpy()
x = (x - np.mean(x, axis = 0)) / np.std(x, axis = 0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print("x_train shape", x_train.shape)
print("x_test", x_test.shape)


## LINEAR REGRESSION MODEL CLASS
class LinearRegression:
    def __init__(self, iterations = 1000, learning_rate=0.01):
        self.W = None
        self.b = None
        self.iterations = iterations
        self.lr = learning_rate
    def fit(self, x, y):
        self.W = np.zeros(x.shape[1]).reshape(x.shape[1],-1)   # (5,1)
        self.b = 0
        for i in range(self.iterations):
            y_pred = np.dot(x, self.W) + self.b       # (1070,1)
            residual = y_pred - y                     # (1070,1)
            dw = (1/x.shape[0]) * np.dot(x.T, residual)    # (5,1070).(1070,1) = (5,1)
            db = (1/x.shape[0]) * np.sum(residual)
            self.W -= self.lr * dw
            self.b -= self.lr * db
            if(i % 1000 == 0):
                mse = np.mean(residual ** 2)
                print(f"MSE for iteration {i} = {mse}")
    def predict(self, x):
        return (np.dot(x, self.W) + self.b).flatten()  # (1,5).(5,1) = (1,1)
    def accuracy(self, x, y):
        y_pred = self.predict(x)
        y = y.flatten()  # <--- Ye line add karo taaki actual y bhi 1D ho jaye
        ssr = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ssr / sst)
    
## FUNCTION CALL AND MAIN FUNCTION
model = LinearRegression(iterations = 10000, learning_rate = 0.001)
print("Training Started ...")
model.fit(x_train, y_train)
print("Training Completed. ")
predictions = model.predict(x_test)
acc = model.accuracy(x_test, y_test)
print(f"\nFinal Model Accuracy (R2 Score): {acc * 100:.2f}%")
while True:
    inp = input("\nPredict insurance cost? (y/n): ").lower()
    if inp != 'y': break
    age = float(input("Age: "))
    sex = 1 if input("Sex (m/f): ").lower() == 'm' else 0
    bmi = float(input("BMI: "))
    kids = float(input("Children: "))
    smoke = 1 if input("Smoker (y/n): ").lower() == 'y' else 0
    user_raw = np.array([age, sex, bmi, kids, smoke])
    user_scaled = (user_raw - np.mean(x, axis = 0)) / np.std(x, axis = 0)
    prediction = model.predict(user_scaled.reshape(1, -1))
    print(f"Estimated Cost: ${prediction[0]:,.2f}")