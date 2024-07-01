# Women's Clothing Review Prediction

## Objective
Predict the ratings of women's clothing based on the reviews using Multinomial Naive Bayes.

## Data Source
- [Kaggle Women's Clothing Reviews Dataset](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

## Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
# Load the dataset
df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')
# Display basic information about the dataset
print(df.info())

# Generate descriptive statistics
print(df.describe())
# Plot the distribution of a categorical column (e.g., Ratings)
sns.countplot(x='Rating', data=df)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
# Handle missing values
df['Review Text'].fillna('', inplace=True)

# Convert categorical ratings into numerical format if necessary
# Example: One-hot encoding (if there are other categorical features)
# df_encoded = pd.get_dummies(df, columns=['categorical_column'])

# Extract features and target variable
X = df['Review Text']
y = df['Rating']

# Feature extraction using TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)
# Target variable (y) is already defined as 'Rating'
# Feature variable (X) is the TF-IDF vectorized 'Review Text'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

report = classification_report(y_test, y_pred)
print(report)
# Example of loading new data and making predictions
new_reviews = ["Great product, very comfortable", "Poor quality, not recommended"]
new_reviews_transformed = vectorizer.transform(new_reviews)
predictions = model.predict(new_reviews_transformed)
print(predictions)

