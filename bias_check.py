import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\Student\Downloads\adult\adult.data", header=None, skipinitialspace=True, na_values=' ?')
df.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
              "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
              "hours-per-week", "native-country", "income"]
df.dropna(inplace=True)

# Encode target
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Encode categorical features (except sex, which we keep as 0/1)
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                    'relationship', 'race', 'native-country']

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Encode sex manually (Female=0, Male=1)
df['sex'] = df['sex'].map({'Female': 0, 'Male': 1})

# Separate features and target
X = df.drop('income', axis=1)
y = df['income']

# Numeric and categorical columns (sex is categorical and kept unscaled)
numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_cols_with_sex = ['sex'] + categorical_cols

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Scale only numeric features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numeric_cols])
X_test_num = scaler.transform(X_test[numeric_cols])

# Combine scaled numeric features and unscaled categorical (including sex)
X_train_final = np.hstack([X_train_num, X_train[categorical_cols_with_sex].values])
X_test_final = np.hstack([X_test_num, X_test[categorical_cols_with_sex].values])

# Get index of sex column in the combined array
sex_col_index = numeric_cols.__len__()  # sex is first in categorical_cols_with_sex

# Train baseline logistic regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final)

print("Base Accuracy:", accuracy_score(y_test, y_pred))

# Accuracy by sex
for gender in [0, 1]:
    idx = X_test_final[:, sex_col_index] == gender
    acc = accuracy_score(y_test[idx], y_pred[idx])
    print(f"{'Female' if gender == 0 else 'Male'} Accuracy: {acc:.4f}")

# Retrain with class_weight balanced for mitigation
model_bal = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model_bal.fit(X_train_final, y_train)
y_pred_bal = model_bal.predict(X_test_final)

print("\nAfter Mitigation (class_weight='balanced'):")
print("Balanced Accuracy:", accuracy_score(y_test, y_pred_bal))

for gender in [0, 1]:
    idx = X_test_final[:, sex_col_index] == gender
    acc = accuracy_score(y_test[idx], y_pred_bal[idx])
    print(f"{'Female' if gender == 0 else 'Male'} Accuracy: {acc:.4f}")
