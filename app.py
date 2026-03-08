import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(page_title="NLP News Classifier", page_icon="📰")

st.title("News Headline Classification")

st.write(
"This AI model classifies news headlines into **World, Sports, Business, and Sci/Tech** categories."
)

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"

    data = pd.read_csv(url, header=None)
    data.columns = ["category", "title", "description"]
    data["text"] = data["title"] + " " + data["description"]
    return data

data = load_data()

# Train model
@st.cache_resource
def train_model(data):

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["category"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english")

    X_train_vec = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)

    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)

    return model, vectorizer, accuracy

model, vectorizer, accuracy = train_model(data)

# Category labels
categories = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

# Sidebar
st.sidebar.title("Model Information")

st.sidebar.write(
"This model predicts the **topic of a news headline** using NLP."
)

st.sidebar.write(
f"**Model Accuracy:** {accuracy*100:.1f}%"
)

st.sidebar.caption(
"Accuracy shows how often the model predicts the correct category."
)

# Example headlines
st.markdown("### Example Headlines")

st.markdown("""
- India wins cricket world cup  
- Apple launches new AI processor  
- Stock markets rise after strong earnings  
- United Nations discusses climate change policy
""")

# Input
headline = st.text_input("Enter a news headline")

# Prediction
if st.button("Predict Category"):

    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:

        headline_vec = vectorizer.transform([headline])

        prediction = model.predict(headline_vec)
        probabilities = model.predict_proba(headline_vec)

        st.success(f"Predicted Category: **{categories[prediction[0]]}**")

        st.subheader("Confidence Scores")

        for i, prob in enumerate(probabilities[0]):
            st.write(f"{categories[i+1]} ({prob*100:.1f}%)")
            st.progress(float(prob))