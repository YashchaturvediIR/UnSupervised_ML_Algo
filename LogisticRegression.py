import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load train and test data
Train_Data = pd.read_csv('Train.csv')
Test_Data = pd.read_csv('Test.csv')

# Create a bag-of-words representation of the content using CountVectorizer
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(Train_Data['content'])
X_test = vectorizer.transform(Test_Data['content'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, Train_Data['title'], test_size=0.2, random_state=42)

# Create and train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")

# Predict on the test set
test_predictions = model.predict(X_test)

# Add predictions to the Test_Data DataFrame
Test_Data['predicted_title'] = test_predictions

# Save the results to a CSV file
Test_Data.to_csv('Test_Results.csv', index=False)
