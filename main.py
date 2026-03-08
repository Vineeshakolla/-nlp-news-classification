import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"

data = pd.read_csv(url, header=None)
data.columns = ["category", "title", "description"]

# Combine title and description
data["text"] = data["title"] + " " + data["description"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["category"], test_size=0.2, random_state=42
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate model
X_test_vec = vectorizer.transform(X_test)
predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Category labels
categories = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

# Interactive prediction loop
while True:

    headline = input("\nEnter a news headline (or type 'exit'): ")

    if headline.lower() == "exit":
        print("Exiting...")
        break

    headline_vec = vectorizer.transform([headline])

    prediction = model.predict(headline_vec)
    probabilities = model.predict_proba(headline_vec)

    print("\nPredicted Category:", categories[prediction[0]])

    print("\nConfidence Scores:")
    for i, prob in enumerate(probabilities[0]):
        print(f"{categories[i+1]}: {prob*100:.2f}%")