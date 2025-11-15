import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report



# Step 1: Read the CSV file into a DataFrame
# Step 2: Define features & target
# Step 3: Encode categorical values
# Step 4: Split train/test
# Step 5: Train model. Make sure Accuracy is > 80%
# Step 6: Evaluate model
# Step 7: Predict new cases
# Step 8: Visualize & interpret


# Read the CSV file into a DataFrame
file_path = '/Users/kishore/Documents/Drexel/Semester-1/Assignment-3/Customer_Review.csv'
df = pd.read_csv(file_path)


# Define features and target variable
target_column = 'Purchased'
#feature_columns = [col for col in df.columns if col != target_column]
feature_columns = [col for col in df.columns if col not in [target_column, 'Serial Number' ]]
print("Feature Columns:", feature_columns)
print("Target Column:", target_column)



# Encode categorical variables
label_encoders = {}
for column in feature_columns + [target_column]:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le



# Split the dataset into training and testing sets
X = df[feature_columns]
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Train the Decision Tree Classifier with hyperparameter tuning
clf = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_split=15, min_samples_leaf=5)
clf.fit(X_train, y_train)
print("Model trained with hyperparameter tuning.")
print("Feature importances:", clf.feature_importances_)



# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Classification report for detailed evaluation
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))




#Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders[target_column].classes_, yticklabels=label_encoders[target_column].classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



# Predict future (new) cases
new_data = pd.DataFrame([
    {'Age': 45, 'Gender': 'Female', 'Review': 'Good', 'Education': 'PG'},
    {'Age': 25, 'Gender': 'Male', 'Review': 'Poor', 'Education': 'UG'}
])


# Visualize & interpret
for col in new_data.columns:
    if col in label_encoders:
        #new_data[col] = label_encoders[col].transform(new_data[col].astype(str))
        new_data[col] = label_encoders[col].transform(new_data[col])

preds = clf.predict(new_data)
pred_labels = label_encoders[target_column].inverse_transform(preds)
print("\nFuture Predictions:")
print(pd.concat([new_data, pd.Series(pred_labels, name='Predicted Purchased')], axis=1))
