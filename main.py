# ======================
# 1. Import Libraries
# ======================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# =========================
# 2. Load the Dataset
# =========================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
df = pd.read_csv(url, names=column_names)

# ===============================
# 3. EDA (on Full Data)
# ===============================
# 3.1 Histograms
df.drop('Outcome', axis=1).hist(bins=20, figsize=(14, 10), layout=(3, 3))
plt.suptitle('Histograms of Features')
plt.show()

# 3.2 Correlation Heatmap
plt.figure(figsize=(10, 7))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Heatmap')
plt.show()

# 3.3 Age Distribution by Outcome
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Age', hue='Outcome', multiple='stack', kde=True, bins=20)
plt.title('Age Distribution by Diabetes Outcome')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 3.4 BMI vs Glucose Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='BMI', y='Glucose', hue='Outcome', palette='Set1')
plt.title('BMI vs Glucose by Outcome')
plt.xlabel('BMI')
plt.ylabel('Glucose')
plt.show()

# 3.5 Outcome Countplot
plt.figure(figsize=(5,4))
sns.countplot(data=df, x='Outcome')
plt.title('Diabetes Outcome Count')
plt.xlabel('Outcome (0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()

# 3.6 Boxplots by Outcome for Key Features
plt.figure(figsize=(15,5))
for i, feature in enumerate(['Glucose', 'BMI', 'Age']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='Outcome', y=feature, data=df)
    plt.title(f'{feature} by Outcome')
plt.tight_layout()
plt.show()

# ============================
# 4. Split Data (Before Scaling or Imputation)
# ============================
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================
# 5. Data Preprocessing (Training Set Stats Only)
# =====================================
cols_with_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero_as_missing:
    # Replace 0 with NaN
    X_train[col] = X_train[col].replace(0, np.nan)
    # Calculate median from TRAINING data only
    median = X_train[col].median()
    # Fill NaN in both train and test with train median
    X_train[col] = X_train[col].fillna(median)
    X_test[col] = X_test[col].replace(0, np.nan)
    X_test[col] = X_test[col].fillna(median)

# Feature scaling: fit on training, apply to both
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================
# 6. Model Training
# ============================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# ============================
# 7. Prediction
# ============================
y_pred = rf.predict(X_test_scaled)

# ============================
# 8. Model Evaluation
# ============================
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# ============================
# 9. Feature Importances
# ============================
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=np.array(column_names[:-1])[indices])
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
