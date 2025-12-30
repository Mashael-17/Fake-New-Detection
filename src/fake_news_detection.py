import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Load the dataset
df = pd.read_csv("train.csv")

# Exploratory Data Analysis (EDA)
print("Dataset Shape:", df.shape)
print("Column Names:", df.columns)
print("First Few Rows:")
print(df.head())

# Handling missing values
df.dropna(subset=['text'], inplace=True)

# Cleaning label column (dropping rows with non 0,1 values)
df = df[df['label'].isin([0, 1])]

# Feature Engineering: Combining title and author
df['combined_text'] = df['title'].fillna('') + " " + df['author'].fillna('') + " " + df['text']


# Improved Text Cleaning Function
def clean_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = word_tokenize(text)  # Tokenization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization and stopword removal
    return " ".join(words)


# Apply text cleaning
df['clean_text'] = df['combined_text'].apply(clean_text)

# Generate and visualize word cloud
plt.figure(figsize=(10, 5))
wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(
    " ".join(df['clean_text']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Most Frequent Words in News Articles')
plt.show()

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

# Convert text into numerical representation
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
}

# Dictionary to store model results
results = []

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]  # Get probability scores

    accuracy = accuracy_score(y_test, y_pred)
    precision = classification_report(y_test, y_pred, output_dict=True)['1']['precision']
    recall = classification_report(y_test, y_pred, output_dict=True)['1']['recall']
    f1 = classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    results.append({"Model": name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1, "ROC AUC": roc_auc})

    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Precision: {precision:.4f}")
    print(f"{name} Recall: {recall:.4f}")
    print(f"{name} F1-Score: {f1:.4f}")
    print(f"{name} ROC AUC: {roc_auc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

# Display model comparison as a table
results_df = pd.DataFrame(results)
print(results_df)

# Plot model comparison
plt.figure(figsize=(10, 5))
results_df.set_index("Model").plot(kind="bar", figsize=(10, 6), colormap="coolwarm")
plt.title('Model Evaluation Metrics Comparison')
plt.xlabel('Model')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.show()
