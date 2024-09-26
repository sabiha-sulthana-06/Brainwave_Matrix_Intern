#Fake News Detection Model

◆Project Overview

This project is part of my internship task, where I built a machine learning model to detect fake news. The dataset contains news articles, and the model classifies them as real or fake.

◆Project Structure

●data:

Contains the dataset and cleaned data files.

◾Files:

▹fake_and_real_news.csv: The original dataset with news articles and their labels.

▹cleaned_fake_and_real_news.csv: The cleaned version of the dataset used for model training.

●models:

Stores the trained models and TF-IDF vectorizer.

◾Files:

▹fake_news_detection_model.pkl: The trained model for fake news detection.

▹tfidf_vectorizer.pkl: The TF-IDF vectorizer used for text feature extraction.

●scripts:

Includes Python scripts for data cleaning, feature extraction, and model training.

◾Files:

▹preprocessing.py: Script for cleaning the text data and extracting features.

▹model_training.py: Script for training the machine learning models.

◆Dataset

The dataset contains two columns:

▹Text: The content of the news article.

▹Label: Indicates if the news is real or fake.

◆Models

◾Logistic Regression

◾Naive Bayes

◆Task Completion

This project fulfills the requirements of building and training a fake news detection model using machine learning.


