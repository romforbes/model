import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 1000

data = {
    'VoterTurnout': np.random.uniform(50,90, n_samples),
    'EconomicGrowth': np.random.normal(2, 0.5, n_samples),
    'UnemploymentRate': np.random.uniform(3, 10, n_samples),
    'Winner': np.random.choice(['PartyA', 'PartyB'], n_samples)
}
election_data = pd.DataFrame({
    'VoterTurnout': np.random.uniform(50, 90, 1000),
    'EconomicGrowth': np.random.normal(2, .5, 1000),
    'UnemploymentRate': np.random.uniform(3, 10, n_samples),
    'Winner': np.random.choice(['PartyA','PartyB'], 1000, replace=True)
})

print(election_data['Winner'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
election_data['Winner'] = label_encoder.fit_transform(election_data['Winner'])

x = election_data[['VoterTurnout', 'EconomicGrowth', 'UnemploymentRate']]
y = election_data['Winner']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)


print(y_train.value_counts())
print(election_data['Winner'].value_counts())

print("y_train distribution:\n", y_train.value_counts())
print("y_test distribution:\n", y_test.value_counts())

print("Class distribution in y_train after stratification:\n", y_train.value_counts())
print("Class distribution in y_test after stratification:\n", y_test.value_counts())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("y_train distribution:\n", y_train.value_counts())
print("y_test distribution:\n", y_test.value_counts())

print("Number of rows in dataset:", len(election_data))
print(election_data['Winner'].value_counts())

print("Class distribution in the entire dataset:\n", y.value_counts())

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

print("Class distribution in y_train after SMOTE:\n", y_train_resampled.value_counts())

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", report)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
