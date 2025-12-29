import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set display options to see all columns on your laptop
pd.set_option('display.max_columns', None)

# 1. Load Data
data = pd.read_csv("diabetic_data.csv")

# 2. Handle missing values (The '?' marks)
data.replace("?", np.nan, inplace=True)

# 3. Drop "Trash" columns (High missing values or useless IDs)
cols_to_drop = ['weight', 'medical_specialty', 'payer_code', 'encounter_id', 'patient_nbr']
data.drop(columns=cols_to_drop, inplace=True)

# 4. Binary Mapping for the Target Variable (High Risk = 1, Others = 0)
data["readmitted"] = data["readmitted"].map({"<30": 1, ">30": 0, "NO": 0})

# 5. Transform Age ranges to numeric mid-points
def age_to_num(age_range):
    cleaned = age_range.replace('[', "").replace(')', "")
    start, end = cleaned.split("-")
    return (int(start) + int(end)) / 2

data["age"] = data["age"].apply(age_to_num)

# 6. One-Hot Encoding for Race and Gender (Dropping first to avoid redundancy)
cols_to_ohe = ["race", "gender"]
data = pd.get_dummies(data, columns=cols_to_ohe, drop_first=True)

# 7. Medication Loop: Convert 24 drugs to Binary (On med = 1, Not on = 0)
meds = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

for col in meds:
    data[col] = data[col].apply(lambda x: 0 if x == "No" else 1)

# 8. Drop Diagnosis columns (Simplifying for first project)
data.drop(columns=["diag_1", "diag_2", "diag_3"], inplace=True)

# 9. Simplify Admission/Discharge IDs (Critical vs. Non-critical)
data["admission_type_id"] = data["admission_type_id"].apply(lambda x: 1 if x == 1 else 0)
data["discharge_disposition_id"] = data["discharge_disposition_id"].apply(lambda x: 1 if x == 1 else 0)
data["admission_source_id"] = data["admission_source_id"].apply(lambda x: 1 if x == 7 else 0)

# 10. Map remaining strings to Binary
data['change'] = data['change'].map({'Ch': 1, 'No': 0})
data['diabetesMed'] = data['diabetesMed'].map({'Yes': 1, 'No': 0})

# --- STEP 11: Map the results ---
result_map = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3, '>7': 2, '>8': 3}
data['max_glu_serum'] = data['max_glu_serum'].map(result_map)
data['A1Cresult'] = data['A1Cresult'].map(result_map)

# --- NEW FIX: Fill NaNs in these two columns specifically ---
# This tells Python: "If the test wasn't taken (NaN), mark it as 0"
data['max_glu_serum'] = data['max_glu_serum'].fillna(0)
data['A1Cresult'] = data['A1Cresult'].fillna(0)

# --- STEP 12: CLEANING ---
# Now dropna will ONLY drop rows where VITAL info like 'race' or 'gender' is missing
data.dropna(inplace=True)

# Now it is safe to convert to int
data['max_glu_serum'] = data['max_glu_serum'].astype(int)
data['A1Cresult'] = data['A1Cresult'].astype(int)

# --- STEP 13: Standardization ---
num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
            'num_medications', 'number_outpatient', 'number_emergency', 
            'number_inpatient', 'number_diagnoses', 'age']

data_mean = data[num_cols].mean(axis=0)
data_std = data[num_cols].std(axis=0)
data[num_cols] = (data[num_cols] - data_mean) / data_std

# Final check for "Unknown" gender
if "gender_Unknown/Invalid" in data.columns:
    data.drop(columns=["gender_Unknown/Invalid"], inplace=True)

data.dropna(inplace=True)

# 14. THE SPLIT (Separating Features from Target)
y = data['readmitted'].values
x = data.drop(columns=['readmitted']).values

print(f"X (Features) shape: {x.shape}")
print(f"y (Target) shape: {y.shape}")
print(f"Number of individual features to learn: {x.shape[1]}")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
x_train = np.array(x_train, dtype=np.float64)
y_train = np.array(y_train, dtype=np.float64)

## Logistic Regression Model
class LogisticRegression:
    def __init__(self, iteration = 1000, learning_rate = 0.01):
        self.iterations = iteration
        self.lr = learning_rate
        self.weights = None
        self.bias = None
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.array(z, dtype=float)))
    def fit(self, x, y):
        row, col = x.shape
        self.weights = np.zeros(col)
        self.bias = 0
        print("Weights and Bias initialised")
        for i in range(self.iterations):
            linear_model = np.dot(x, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = (1/row) * np.dot(x.T, (y_pred - y))
            db = (1/row) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            if i % (self.iterations/10) == 0:
                print(f"Iteration {i}: Weights are updating...")
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    def get_metrics(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Confusion Matrix": [[tn, fp], [fn, tp]]
        }


# Initialize and Train
model = LogisticRegression(iteration=5000, learning_rate=0.1)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
results = LogisticRegression.get_metrics(y_test, predictions)
print("\n--- Final Model Performance ---")
for metric, value in results.items():
    print(f"{metric}: {value}")
while True:
    choice = input("\nDo you want to predict for a new patient? (yes/no): ").lower().strip()
    if choice == 'no':
        print("Exiting system. Goodbye!")
        break
    if choice == 'yes':
        try:
            print("\nPlease enter the following details:")
            user_features = np.zeros(x.shape[1])
            feature_names = data.drop(columns=['readmitted']).columns.tolist()
            age_val = float(input("Age (numeric): "))
            time_val = float(input("Time in hospital (days): "))
            lab_val = float(input("Number of lab procedures: "))
            med_val = float(input("Number of medications: "))
            user_features[feature_names.index('age')] = (age_val - data_mean['age']) / data_std['age']
            user_features[feature_names.index('time_in_hospital')] = (time_val - data_mean['time_in_hospital']) / data_std['time_in_hospital']
            user_features[feature_names.index('num_lab_procedures')] = (lab_val - data_mean['num_lab_procedures']) / data_std['num_lab_procedures']
            user_features[feature_names.index('num_medications')] = (med_val - data_mean['num_medications']) / data_std['num_medications']
            gender = input("Gender (Male/Female): ").strip().capitalize()
            if gender == "Male" and "gender_Male" in feature_names:
                user_features[feature_names.index('gender_Male')] = 1
            final_input = user_features.reshape(1, -1)
            prob = model.sigmoid(np.dot(final_input, model.weights) + model.bias)[0]
            print(f"\n--- PREDICTION RESULT ---")
            print(f"Readmission Probability: {prob:.2%}")
            if prob > 0.5:
                print("Conclusion: HIGH RISK of readmission.")
            else:
                print("Conclusion: LOW RISK of readmission.")
        except Exception as e:
            print(f"Error in input: {e}. Please try again.")
    else:
        print("Invalid input. Please type 'yes' or 'no'.")