# Phase_4_Project

# Sentiment Analysis Project

This project demonstrates the process of building a **Sentiment Analysis** model on a dataset of tweets related to Apple and Google products. The objective is to classify the sentiment of tweets into three categories: **positive**, **neutral**, and **negative**. This project covers multiple stages of data analysis, from understanding the business problem to preparing data, training machine learning models, evaluating their performance, and visualizing the results.

## Project Overview
S
- **Title**: Sentiment Analysis on Tweets
- **Language**: Python
- **Libraries Used**:
  - **Pandas**: For data manipulation and analysis.
  - **NumPy**: For numerical operations.
  - **Matplotlib/Seaborn**: For data visualization.
  - **NLTK**: For text preprocessing and analysis.
  - **Scikit-learn**: For machine learning models and evaluation.
  - **Imbalanced-learn**: For handling class imbalances.
  - **Transformers**: For advanced NLP techniques using pre-trained models.
  - **Joblib**: For saving and loading models.
  - **Torch**: For deep learning (if applicable).

## 1. Business Understanding

### 1.1 Introduction

This project aims to classify sentiments in tweets mentioning **Apple** and **Google** products into three categories: **positive**, **neutral**, or **negative**. Sentiment analysis is a key application of natural language processing (NLP) that helps organizations understand public opinion at scale.

### 1.2 Problem Statement

Apple and Google heavily rely on the perception of their customers towards their products. As customers continue to provide feedback on platforms like **Twitter (X)**, it’s crucial to analyze the sentiment of this feedback. This project aims to build a model that helps classify tweets based on the sentiment they express.

### 1.3 Objectives

- **Main Objective**: Build an accurate NLP model to classify tweets mentioning Apple and Google products into sentiment categories: **positive**, **neutral**, and **negative**.
  
- **Specific Objectives**:
  - Explore and analyze tweet data to provide actionable insights.
  - Preprocess the data using NLP techniques like tokenization and lemmatization.
  - Evaluate the performance of the classifiers using appropriate metrics (e.g., accuracy, confusion matrix, F1 score).

## 2. Data Understanding

### 2.1 Initial Data Loading and Inspection

The dataset is loaded using **pandas**, and the initial inspection reveals that it contains tweets along with their respective sentiment labels. The dataset is provided in a CSV file format (`judge-1377884607_tweet_product_company.csv`).

### 2.2 Data Overview

After loading the data, the following commands provide insight into the dataset:

df.head()  # Show first few rows of the dataset
df.info()  # Dataset information (types and nulls)
df.describe()  # Statistical summary of the dataset
df.shape  # Dimensions of the dataset

## 3. Data Preprocessing

In this project, the text data undergoes several preprocessing steps, including:

- **Text cleaning**: Removing special characters and irrelevant information.
- **Tokenization**: Breaking down the text into individual words.
- **Removing stop words**: Filtering out common but non-informative words (e.g., "the", "and").
- **Lemmatization**: Reducing words to their base or root form.

### 3.1 Word Cloud Visualization

A **WordCloud** is generated to visualize the most frequent terms in the dataset. This helps in understanding which words are most commonly used in tweets related to Apple and Google.

## 4. Model Building

### 4.1 Model Selection

Multiple machine learning models were trained, including:

- **Logistic Regression**: A linear model for binary classification.
- **Linear Support Vector Classifier (SVC)**: A robust model for text classification.
- **K-Nearest Neighbors (KNN)**: A non-parametric model for classification.

### 4.2 Data Splitting and Training

The dataset is split into training and testing sets using `train_test_split` from **sklearn**. The models are then trained on the training set and tested on the test set.

### 4.3 Handling Class Imbalance

Since the dataset has imbalanced classes, techniques like **RandomOverSampler** from **imblearn** are used to oversample the minority class and balance the class distribution.

## 5. Model Evaluation

### 5.1 Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy**: The proportion of correctly classified tweets.
- **Confusion Matrix**: A matrix showing the true positive, false positive, true negative, and false negative counts.
- **Classification Report**: A detailed report including precision, recall, and F1-score for each class.

### 5.2 Performance Results

The models were evaluated on the test set, and their performances were compared. The results showed that the **LinearSVC** model performed the best in terms of F1-score, closely followed by **Logistic Regression**.

Example of evaluation for the **LinearSVC** model:

classification_report(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()

## 6. Advanced Techniques (Optional)

For further improvement, **pre-trained deep learning models** from the **Transformers** library were used to enhance sentiment classification accuracy. These models provide a powerful approach to NLP tasks, leveraging state-of-the-art transformer architectures like **BERT**.

## 7. Conclusion

The project successfully developed a sentiment analysis model for classifying tweets about Apple and Google products. By using **Logistic Regression**, **LinearSVC**, and **KNN**, the model was able to predict the sentiment of tweets with reasonable accuracy. Future work could involve fine-tuning the pre-trained transformer models to achieve even better performance.

## Dataset

The dataset contains tweets with text about Apple and Google products. The labels are classified into three categories: **positive**, **neutral**, and **negative**. This dataset is essential for training and testing the sentiment analysis models.
