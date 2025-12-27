## IMPORTING NCESSARY LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

## DATA PREPROCESSING

## READING FILES
data = pd.read_csv("yield_df.csv")
data.drop("Unnamed: 0", axis = 1, inplace = True)
all_items = sorted(data['Item'].unique())
all_areas = sorted(data['Area'].unique())

## CONVERTING CATEGORICAL DATA TO NUMERICAL
data_onehot = pd.get_dummies(data, columns = ["Area", "Item"], prefix = ["Country", "Crop"], drop_first = True)
data_onehot = data_onehot.astype(int)

## SEPERATING FEATURES FROM TARGET
y = data_onehot["hg/ha_yield"]
x = data_onehot.drop("hg/ha_yield", axis = 1)

## SPLITTING DATA INTO TRAIN AND TEST DATA
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

## STANDARDIZING THE VALUES
num_col = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Year"]
train_mean = x_train[num_col].mean()
train_std = x_train[num_col].std()
x_train[num_col] = (x_train[num_col] - train_mean) / train_std
x_test[num_col] = (x_test[num_col] - train_mean) / train_std

## GENERATING THE POLYNOMIAL COLUMNS 
poly_num = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Year"]
x_train_num = x_train[poly_num].values
x_test_num = x_test[poly_num].values
x_train_num_sq = x_train_num ** 2
x_test_num_sq = x_test_num ** 2
x_train_poly = np.hstack([x_train.values, x_train_num_sq])
x_test_poly = np.hstack([x_test.values, x_test_num_sq])

## POLYNOMIAL REGRESSION CLASS
class polynomial_regression:
    def __init__(self):
        self.theta = None
    def add_bias(self, x):
        return np.hstack([np.ones((x.shape[0], 1)), x])
    def fit(self, x, y):
        x = self.add_bias(x)
        self.theta = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
    def predict(self, x):
        x = self.add_bias(x)
        return np.dot(x, self.theta)
    def mse(self, y, y_pred):
        return (1/y.shape[0]) * np.sum((y - y_pred) ** 2)
    def mae(self, y, y_pred):
        return (1/y.shape[0]) * np.sum(abs(y - y_pred))
    def rsq(self, y, y_pred):
        ssr = np.sum((y - y_pred) ** 2)
        y_mean = np.mean(y)
        sst = np.sum((y - y_mean) ** 2)
        return 1 - (ssr / sst)

## MAIN FUNCTION
model = polynomial_regression()
model.fit(x_train_poly, y_train)
test_preds = model.predict(x_test_poly)
print("--- MODEL TRAINING COMPLETE ---")
print(f"Mean Absolute Error: {model.mae(y_test.values, test_preds):.2f}")
print(f"R-Squared Score:     {model.rsq(y_test.values, test_preds):.4f}")
print("-------------------------------\n")

## USER INPUT
while True:
    choice = input("Do you want to predict yield for a custom input? (y/n): ").lower()
    if choice != 'y':
        break
    try:
        print("\nAvailable Crops:", ", ".join(all_items[:5]), "...")
        item = input("Enter Crop (Item): ")
        area = input("Enter Country (Area): ")
        year = float(input("Enter Year: "))
        rain = float(input("Enter Average Rainfall (mm): "))
        pest = float(input("Enter Pesticides (tonnes): "))
        temp = float(input("Enter Average Temp (C): "))
        user_input_df = pd.DataFrame(0, index=[0], columns=x.columns)
        user_input_df['Year'] = year
        user_input_df['average_rain_fall_mm_per_year'] = rain
        user_input_df['pesticides_tonnes'] = pest
        user_input_df['avg_temp'] = temp
        country_col = f"Country_{area}"
        crop_col = f"Crop_{item}"
        if country_col in user_input_df.columns:
            user_input_df[country_col] = 1
        if crop_col in user_input_df.columns:
            user_input_df[crop_col] = 1
        user_input_df[num_col] = (user_input_df[num_col] - train_mean) / train_std
        user_final_num = user_input_df[num_col].values
        user_final_poly = np.hstack([user_input_df.values, user_final_num**2])
        prediction = model.predict(user_final_poly)
        print(f"\n>>> PREDICTED YIELD: {prediction[0]:.2f} hg/ha\n")
    except Exception as e:
        print(f"Error: {e}. Please ensure inputs match dataset format.")