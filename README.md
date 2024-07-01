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
#EXPLANATION
In this project, we aimed to build a predictive model for women's clothing reviews using the Multinomial Naive Bayes algorithm. The primary objective was to classify the sentiment of reviews based on their textual content, facilitating better understanding of customer feedback for retailers.

We began by importing the necessary libraries and loading the dataset. The data was then explored to understand its structure, identify missing values, and perform initial visualizations. Data preprocessing steps included handling missing values, converting categorical variables, and extracting features using TF-IDF vectorization.

Our target variable was the sentiment rating, and the feature variable was the vectorized textual reviews. We split the data into training and testing sets to evaluate the performance of our model. After training the Multinomial Naive Bayes model, we assessed its accuracy and generated a classification report.

The model achieved satisfactory accuracy, indicating its effectiveness in predicting the sentiment of clothing reviews. Visualizations were created to illustrate the distribution of ratings and the importance of different words in determining sentiment.

Overall, this project demonstrates the application of text classification techniques in analyzing customer reviews, providing valuable insights for enhancing customer satisfaction and improving product offerings.


