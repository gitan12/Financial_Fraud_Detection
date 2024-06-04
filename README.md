Here's a structured summary of the PaySim data exploration and model evaluation process in Python:

### PaySim Dataset Overview
The PaySim simulator creates synthetic datasets to mimic real-world mobile money transactions, facilitating fraud detection research without compromising sensitive information. The dataset includes transaction types like CASH-IN, CASH-OUT, DEBIT, PAYMENT, and TRANSFER, with indicators for fraudulent activity and flagged frauds.

### Libraries and Data Loading
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/PS_20174392719_1491204439457_log.csv')
```

### Data Exploration
#### Display the first 10 rows and dataset shape
```python
df.head(10)
df.shape
# Output: (6362620, 11)
```

#### Data Types and Missing Values
```python
df.info()
df.describe()
```

### Exploratory Data Analysis (EDA)
#### Count Plot of Fraudulent Transactions
```python
sns.countplot(x='isFraud', data=df)
plt.show()
```

#### Distribution of Transaction Types
```python
df['type'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Types')
plt.show()
```

#### Scatter Plot: Amount vs. Old Balance Origin
```python
sns.scatterplot(x='amount', y='oldbalanceOrg', data=df)
plt.title('Scatter Plot: Amount vs. Old Balance Origin')
plt.show()
```

### Preprocessing
#### Check for Missing Values
```python
df.isnull().sum()
```

#### Encode Categorical Variables
```python
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
```

#### Correlation Matrix
```python
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
```

#### Sampling the Data
```python
sample_df = df.sample(frac=0.4, random_state=42)
```

#### Split Data into Features and Target
```python
X = sample_df.drop(['isFraud', 'isFlaggedFraud', 'nameDest', 'nameOrig'], axis=1)
y = sample_df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Model Training and Hyperparameter Tuning
#### Logistic Regression
```python
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
```

#### Decision Tree
```python
dt = DecisionTreeClassifier()
dt_grid = {'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10]}
dt_clf = GridSearchCV(dt, dt_grid, cv=3)
dt_clf.fit(X_train, y_train)
```

#### Random Forest
```python
rf = RandomForestClassifier()
rf_grid = {'n_estimators': [1, 3], 'max_depth': [10, 20], 'min_samples_leaf': [1, 2]}
rf_clf = GridSearchCV(rf, rf_grid, cv=3)
rf_clf.fit(X_train, y_train)
```

### Model Evaluation
#### Evaluation Function
```python
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Accuracy:", accuracy_score(y_test, y_pred))

    auc = roc_auc_score(y_test, y_pred_proba)
    print("AUC Score:", auc)
```

#### Evaluating Each Model
```python
print("Logistic Regression Results:")
evaluate_model(log_reg, X_test, y_test)

print("Decision Tree Results:")
evaluate_model(dt_clf, X_test, y_test)

print("Random Forest Results:")
evaluate_model(rf_clf, X_test, y_test)
```

### Summary of Results
- **Logistic Regression:** High accuracy (~99.82%), AUC score: 0.8555
- **Decision Tree:** High accuracy (~99.97%), AUC score: 0.9631
- **Random Forest:** High accuracy (~99.98%), AUC score: 0.9847

## Conclusion

In this project, we analyzed the PaySim synthetic dataset to detect fraudulent transactions in mobile money services. 

#### Data Exploration
- **Dataset Overview**: The dataset contains 6,362,620 rows and 11 columns, with a mix of transaction types, amounts, and customer balances.
- **Imbalance in Data**: Only 0.13% of the transactions are fraudulent, highlighting the challenge of detecting rare fraud cases.

#### Exploratory Data Analysis (EDA)
- **Fraud Distribution**: Fraudulent transactions predominantly occur in the CASH-OUT and TRANSFER categories.
- **Outliers and Spread**: Significant variability and outliers were noted in transaction amounts and balances, especially in non-fraudulent cases.

#### Feature Engineering
- **Categorical Encoding**: Transaction types were label encoded to facilitate model training.
- **Correlation Analysis**: Correlation matrix heatmaps helped visualize relationships between features.

#### Sampling and Preprocessing
- **Sampling**: A 40% sample of the dataset was used to reduce computational load.
- **Scaling**: StandardScaler was used to normalize feature values, ensuring consistent model input.

#### Model Training and Evaluation
- **Logistic Regression**: Achieved an accuracy of 99.82% and an AUC score of 0.86, but struggled with recall for fraudulent transactions.
- **Decision Tree**: Performed better with an accuracy of 99.97% and an AUC score of 0.96, showing improved recall.
- **Random Forest**: Similar performance to the Decision Tree with an accuracy of 99.97% and an AUC score of 0.96, demonstrating robustness in fraud detection.

### Key Insights
- **Fraud Detection Models**: Both Decision Tree and Random Forest classifiers outperformed Logistic Regression in detecting fraudulent transactions.
- **Imbalanced Data Handling**: The models handled the class imbalance effectively, but further techniques like SMOTE or undersampling could enhance performance.
- **Feature Importance**: Transaction type, amount, and balances were critical features in identifying fraudulent activities.


Random Forest performed the best with the highest AUC score and accuracy, indicating robust performance in detecting fraudulent transactions in the PaySim dataset.
