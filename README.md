# NLP News Headline Classification

## Live Application

The deployed application can be accessed at:

https://ag-news-nlp-classifier.streamlit.app

---

## Overview

This project implements a Natural Language Processing (NLP) pipeline to classify news headlines into topic categories. The system uses TF-IDF feature extraction combined with a Multinomial Naive Bayes classifier to predict the category of a news headline.

The project also includes an interactive web interface built with Streamlit, allowing users to enter headlines and view predicted categories along with confidence scores.

---

## Categories

The model classifies news headlines into the following categories:

* World
* Sports
* Business
* Sci/Tech

---

## Dataset

The project uses the AG News Classification Dataset.
It contains over 120,000 news samples collected from various news sources.

Each record includes:

* News Title
* News Description
* Category Label

To simplify deployment and avoid storing large files in the repository, the dataset is loaded directly from an online source at runtime.

---

## Methodology

The machine learning pipeline consists of the following steps:

1. Dataset loading from an external source
2. Text preprocessing and feature construction
3. Combining headline and description text
4. Train–test dataset split
5. TF-IDF vectorization for text representation
6. Model training using Multinomial Naive Bayes
7. Model evaluation using classification accuracy
8. Prediction using both a command-line interface and a web interface

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
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```
git clone https://github.com/YOUR_GITHUB_USERNAME/nlp-news-classification.git
```

Navigate to the project directory:

```
cd nlp-news-classification
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Application

Run the Streamlit application:

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

Typical classification accuracy on the test set is approximately **90%**.

---

## Possible Extensions

Future improvements may include:

* Comparing additional models such as Logistic Regression or Support Vector Machines
* Applying advanced NLP techniques such as transformer-based architectures
* Improving text preprocessing and feature engineering
* Deploying the system as a scalable production service

---

## Author

Kolla Vineesha

Undergraduate student in Computer Science with interests in Artificial Intelligence and Natural Language Processing.
