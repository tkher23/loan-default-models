import pandas as pd
import chardet
import pandas as pd
import tensorflow
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



#Load Data

# Function to try reading CSV with different encodings
def try_read_csv(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f'Successfully read the file with {enc} encoding')
            return df
        except Exception as e:
            print(f'Failed with {enc} encoding: {e}')
    raise Exception('Failed to read the file with all attempted encodings')

# Attempt to read the CSV files with different encodings
df1 = try_read_csv('sba-2010to2019.csv')
df2 = try_read_csv('sba-2020present.csv')  # Replace 'another_file.csv' with your second CSV file

# Select relevant columns that are identical in both files
columns_to_select = ["BorrCity", "BorrState", "BorrZip", "BankName", "GrossApproval", 
                     "InitialInterestRate", "FixedOrVariableInterestInd", "TermInMonths", 
                     "NaicsCode", "BusinessType", "LoanStatus"]

df1_selected = df1[columns_to_select]
df2_selected = df2[columns_to_select]

# Combine the DataFrames
combined_df = pd.concat([df1_selected, df2_selected], ignore_index=True)

# Filter rows where LoanStatus is "PIF" or "CHGOFF"
combined_df = combined_df[combined_df['LoanStatus'].isin(['PIF', 'CHGOFF'])]

# Convert LoanStatus to binary values
combined_df['LoanStatus'] = combined_df['LoanStatus'].map({'CHGOFF': 1, 'PIF': 0})


# Save the combined DataFrame to a new CSV file
combined_df.to_csv('combined_file.csv', index=False)

# Display the first few rows of the combined DataFrame
print(combined_df.head())


# Handle Missing Values - using mean imputation for numerical columns

imputer = SimpleImputer(strategy='mean')
combined_df[["GrossApproval", "InitialInterestRate", "TermInMonths"]] = imputer.fit_transform(combined_df[["GrossApproval", "InitialInterestRate", "TermInMonths"]])

# For categorical columns, we can fill missing values with a placeholder or mode
combined_df["BorrCity"].fillna("Unknown", inplace=True)
combined_df["BorrState"].fillna("Unknown", inplace=True)
combined_df["BorrZip"].fillna("00000", inplace=True)
combined_df["BankName"].fillna("Unknown", inplace=True)
combined_df["FixedOrVariableInterestInd"].fillna("Unknown", inplace=True)
combined_df["NaicsCode"].fillna("000000", inplace=True)
combined_df["BusinessType"].fillna("Unknown", inplace=True)
combined_df["LoanStatus"].fillna("Unknown", inplace=True)

# Normalizing Numerical Data

scaler = MinMaxScaler()
combined_df[["GrossApproval", "InitialInterestRate", "TermInMonths"]] = scaler.fit_transform(combined_df[["GrossApproval", "InitialInterestRate", "TermInMonths"]])

# One-Hot Encoding for Low Cardinality Nominal Variables - Fixed Or Variable, BusinessType
one_hot_cols = ['FixedOrVariableInterestInd', 'BusinessType']
combined_df = pd.get_dummies(combined_df, columns=one_hot_cols, drop_first=True)

# Target Encoding for High Cardinality Nominal Variables - BorrCity, BorrState, BorrZip, BankName, NaicsCode
target_encoder = TargetEncoder()
combined_df['NaicsCode_TargetEncoded'] = target_encoder.fit_transform(combined_df['NaicsCode'], combined_df['LoanStatus'])
combined_df.drop('NaicsCode', axis=1, inplace=True)


combined_df['BorrCity_TargetEncoded'] = target_encoder.fit_transform(combined_df['BorrCity'], combined_df['LoanStatus'])
combined_df.drop('BorrCity', axis=1, inplace=True)


combined_df['BorrState_TargetEncoded'] = target_encoder.fit_transform(combined_df['BorrState'], combined_df['LoanStatus'])
combined_df.drop('BorrState', axis=1, inplace=True)

combined_df['BorrZip_TargetEncoded'] = target_encoder.fit_transform(combined_df['BorrZip'], combined_df['LoanStatus'])
combined_df.drop('BorrZip', axis=1, inplace=True)

combined_df['BankName_TargetEncoded'] = target_encoder.fit_transform(combined_df['BankName'], combined_df['LoanStatus'])
combined_df.drop('BankName', axis=1, inplace=True)

# Prepare Features and Target Variables 
X = combined_df.drop('LoanStatus', axis=1)
Y = combined_df['LoanStatus']


# Split data into 60% training, 20% cross-validation, and 20% test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Training set size:", len(X_train))
print("Cross Validation set size:", len(X_val))
print("Test set size:", len(X_test))

# Creation of competing neural network models
first = X.shape[1]
print("Number of variables:", first)

def create_model(units_list, input_shape=(X_train.shape[1],), activation='relu'):
    model = Sequential()
    model.add(Dense(units_list[0], input_shape=input_shape, activation=activation))
    for units in units_list[1:]:
        model.add(Dense(units, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(0.001),
        loss=tensorflow.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    return model

# Define the structures to test
structures = [
    [64],
    [64, 32],
    [64, 32, 16],
    [64, 32, 16, 8],
    [128, 64],
    [128, 64, 32]
]

# Store metrics
results = []

# Train and evaluate models for each structure
for structure in structures:
    print(f"Training model with structure: {structure}")
    model = create_model(structure)
    model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)  # Reduced epochs for quicker testing

    # Evaluate on validation set
    y_val_pred = (model.predict(X_val) > 0.5).astype("int32").ravel()
    accuracy_val = accuracy_score(y_val, y_val_pred)

    # Evaluate on training set
    y_train_pred = (model.predict(X_train) > 0.5).astype("int32").ravel()
    accuracy_train = accuracy_score(y_train, y_train_pred)

    # Evaluate on test set
    y_test_pred = (model.predict(X_test) > 0.5).astype("int32").ravel()
    accuracy_test_nn = accuracy_score(y_test, y_test_pred)

    results.append({
        'structure': structure,
        'accuracy_train': accuracy_train,
        'accuracy_val': accuracy_val,
        'accuracy_test': accuracy_test_nn,
        'model': model
    })

# Convert results to DataFrame for easy comparison
results_df = pd.DataFrame(results)
print(results_df)

# Select the best model based on validation accuracy
best_model_info = results_df.loc[results_df['accuracy_val'].idxmax()]
best_model_nn = best_model_info['model']
print(f"Best Neural Network Model: {best_model_info['structure']} layers")
print(f"Train Accuracy: {best_model_info['accuracy_train']}")
print(f"Validation Accuracy: {best_model_info['accuracy_val']}")
print(f"Test Accuracy: {best_model_info['accuracy_test']}")


# --- Decision Tree Logic + Random Forest Parameters ---
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'criterion': ['entropy'],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 5, 10],
    'max_features': ['log2']
}


# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Evaluate the best model on the validation set
best_rf = grid_search.best_estimator_

# Evaluate on training set
y_train_pred = best_rf.predict(X_train)
accuracy_train = accuracy_score(y_train, y_train_pred)
print(f"Train Accuracy: {accuracy_train}")
print("Train Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
print("Train Classification Report:\n", classification_report(y_train, y_train_pred))

# Evaluate on validation set
y_val_pred = best_rf.predict(X_val)
accuracy_val = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy with Best Model: {accuracy_val}")
print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))

# Evaluate on test set
y_test_pred = best_rf.predict(X_test)
accuracy_test_rf = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy with Best Model: {accuracy_test_rf}")
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Test Classification Report:\n", classification_report(y_test, y_test_pred))

# Check for overfitting
if accuracy_train > accuracy_test_rf and accuracy_train - accuracy_test_rf > 0.05:
    print("The model is likely overfitting.")
else:
    print("The model does not seem to be overfitting.")

# Select final model
if  accuracy_test_rf > best_model_info['accuracy_test']:
    final_model = best_rf
    print(f'Random Forest is the final model. With the parameters {grid_search.best_params}, this model had {accuracy_test_rf} accuracy.'_)
else:
    final_model = best_model_nn
    print(f'Neural Network with params {best_model_info['structure']} is the final model. This model had {accuracy_test_nn} accuracy.')
