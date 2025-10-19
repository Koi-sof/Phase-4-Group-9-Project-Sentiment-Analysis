# Phase_4_Project

# Sentiment Analysis Project

This project demonstrates the process of building a **Sentiment Analysis** model on a dataset of tweets related to Apple and Google products. The objective is to classify the sentiment of tweets into three categories: **positive**, **neutral**, and **negative**. This project covers multiple stages of data analysis, from understanding the business problem to preparing data, training machine learning models, evaluating their performance, and visualizing the results.

## Project Overview

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

**Main Objective**: Build an accurate NLP model to classify tweets mentioning Apple and Google products into sentiment categories: **positive**, **neutral**, and **negative**.

 **Specific Objectives**:
 
- To explore and analyze the tweet data between the two companies

- To preprocess the data using Natural Language Processing techniques.
    
- To develop and evaluate classifier performance using appropriate metrics(e.g., accuracy, confusion matrix, F1 score).

- To provide actionable, data-driven insights and recommendations

- To successfully deploy the model into the production environment


## 2. Data Understanding

The dataset contains tweets with text about Apple and Google products. The sentiment labels are classified into three categories: **positive**, **neutral**, and **negative**. This dataset is essential for training and testing the sentiment analysis models.

### 2.1 Important Visualizations

#### 2.1.1 Distribution of Sentiment Class

<img width="631" height="483" alt="image" src="https://github.com/user-attachments/assets/bd6c0282-ce3f-490b-a400-4b2632d66d89" />

#### 2.1.2 Distribution of Sentiment Across Google and Apple

<img width="971" height="580" alt="image" src="https://github.com/user-attachments/assets/83c41a1f-db48-4fd8-b31e-c42048372a6b" />

#### 2.1.3 Word Cloud of Tweets

<img width="792" height="431" alt="image" src="https://github.com/user-attachments/assets/fd08056f-61db-42aa-b800-c4bb6b9c1930" />

#### 2.1.4 Classification Matrix of the Best Model

<img width="632" height="475" alt="image" src="https://github.com/user-attachments/assets/5f4763d2-c14f-4f3d-bd30-64e511d3b170" />


## 3. Data Preprocessing

In this project, the text data undergoes several preprocessing steps, including:

- **Text cleaning**: Removing special characters and irrelevant information.
- **Tokenization**: Breaking down the text into individual words.
- **Removing stop words**: Filtering out common but non-informative words (e.g., "the", "and").
- **Lemmatization**: Reducing words to their base or root form.

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

The `LinearSVC` model was trained again with balanced classes and n-grams and achieved the best results across all three classes.

SVM with N-grams and Balanced Classes Results:

```
                  precision    recall  f1-score   support

Negative emotion       0.90      0.99      0.94      1106
 Neutral emotion       0.78      0.69      0.73      1106
Positive emotion       0.78      0.79      0.78      1107

        accuracy                           0.82      3319
       macro avg       0.82      0.82      0.82      3319
    weighted avg       0.82      0.82      0.82      3319
```

## 6. Advanced Techniques (Optional)

For further improvement, **pre-trained deep learning models** from the **Transformers** library were used to enhance sentiment classification accuracy. These models provide a powerful approach to NLP tasks, leveraging state-of-the-art transformer architectures like **BERT**.

## 7. Conclusion

The project successfully developed a sentiment analysis model for classifying tweets about Apple and Google products. By using **Logistic Regression**, **LinearSVC**, and **KNN**, the model was able to predict the sentiment of tweets with reasonable accuracy. Future work could involve fine-tuning the pre-trained transformer models to achieve even better performance.
