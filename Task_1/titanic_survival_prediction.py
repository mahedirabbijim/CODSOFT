
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Drop columns that may not help much or have too many missing values
data = data.drop(columns=['Cabin', 'Name', 'Ticket'])

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Convert categorical columns to numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Define features and target
X = data.drop(columns=['Survived'])
y = data['Survived']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Optional: Predict for a new passenger (just for test)
# new_passenger = pd.DataFrame([{
#     'Pclass': 3, 'Sex': 0, 'Age': 22, 'SibSp': 1, 'Parch': 0, 'Fare': 7.25, 'Embarked': 0
# }])
# print("Survived" if model.predict(new_passenger)[0] == 1 else "Did not survive")
