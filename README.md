# NLP News Headline Classification

## Overview

This project implements a Natural Language Processing (NLP) pipeline to classify news headlines into topic categories.
The system uses TF-IDF feature extraction and a Multinomial Naive Bayes classifier to predict the category of a news headline.

The project also includes an interactive web application built with Streamlit that allows users to input a news headline and receive a predicted category along with confidence scores.

---

## Categories

The model classifies headlines into the following categories:

* World
* Sports
* Business
* Sci/Tech

---

## Dataset

This project uses the **AG News Classification Dataset**, which contains more than 120,000 news samples across four categories.

Each record contains:

* News Title
* News Description
* Category Label

The dataset is **automatically downloaded from a public source when the application runs**, so it is not stored directly in the repository.

---

## Methodology

The machine learning pipeline follows these steps:

1. Dataset loading
2. Text preprocessing (combining title and description)
3. Train–test dataset split
4. TF-IDF vectorization
5. Model training using Multinomial Naive Bayes
6. Model evaluation using classification accuracy
7. Prediction through a command-line interface and a Streamlit web application

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit

---

## Project Structure

```
nlp-news-classification
│
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/Vineeshakolla/nlp-news-classification.git
```

Navigate to the project directory:

```
cd nlp-news-classification
```

Install the required dependencies:

```
pip install -r requirements.txt
```

---

## Running the Application

Run the Streamlit web application:

```
streamlit run app.py
```

Then open the following address in your browser:

```
http://localhost:8501
```

---

## Example Prediction

Input headline:

```
India wins cricket world cup
```

Predicted category:

```
Sports
```

---

## Model Performance

Model: Multinomial Naive Bayes
Feature Extraction: TF-IDF

Typical accuracy on the test set: **approximately 90%**.

---

## Possible Extensions

Potential future improvements include:

* Comparing additional machine learning models such as Logistic Regression or Support Vector Machines
* Incorporating modern NLP models such as transformer-based architectures
* Improving preprocessing and feature engineering techniques
* Deploying the application as a publicly accessible web service

---

## Author

Kolla Vineesha
Undergraduate student in Computer Science with interests in Artificial Intelligence, Machine Learning, and Natural Language Processing.
