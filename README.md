# 📰 Fake News Detector

An AI-powered Fake News Detection System that uses Natural Language Processing (NLP) and Machine Learning to classify news articles as **REAL** or **FAKE**.

---

## 📌 Project Overview

Fake news has become a major problem with the rapid growth of digital media and social networking platforms.

This project uses Machine Learning and Natural Language Processing techniques to analyze the textual content of a news article and predict whether it is likely to be **REAL** or **FAKE**.

The system provides an interactive web interface built using Streamlit.

---

## ✨ Features

- 📰 Real-time fake news prediction
- 🤖 Machine Learning based classification
- 🧹 NLP text preprocessing
- 📊 TF-IDF text vectorization
- 🔍 Multiple ML models for comparison
- 📈 Accuracy, Precision, Recall and F1-Score evaluation
- 🎯 Prediction confidence
- 📋 Sample news input
- 🗑️ Clear input option
- 📜 Recent prediction history
- 💻 Interactive Streamlit web interface

---

## 🧠 Machine Learning Models

The project evaluates multiple machine learning algorithms:

1. **Logistic Regression**
2. **Multinomial Naive Bayes**
3. **Linear SVM**

The model with the best **F1-Score** is selected and saved for prediction.

---

## 🔄 Project Workflow

```text
News Article
     ↓
Text Preprocessing
     ↓
Lowercase Conversion
     ↓
URL & Punctuation Removal
     ↓
Stopword Removal
     ↓
Stemming
     ↓
TF-IDF Vectorization
     ↓
Machine Learning Model
     ↓
REAL / FAKE Prediction
     ↓
Confidence Score