# Final Project Report: Sentiment Analysis on Reviews using Deep Learning

## 1\. Introduction

Sentiment analysis is a fundamental task in Natural Language Processing (NLP) that aims to determine the emotional tone behind a body of text. This project focuses on classifying sentences from user reviews into binary sentiment categories: **Positive (1)** or **Negative (0)**. The ability to automatically gauge sentiment is crucial for businesses monitoring brand reputation and product feedback.

## 2\. Objectives

The primary objectives of this project are:

1. To implement a robust text preprocessing pipeline using NLP techniques.
2. To develop and train three distinct machine learning models: a Dense Neural Network (DNN), a Vanilla Recurrent Neural Network (RNN), and a Long Short-Term Memory (LSTM) network.
3. To establish a baseline using a Dummy Classifier.
4. To compare the performance of these models using metrics such as Accuracy, F1-Score, and Cohen’s Kappa.

## 3\. Dataset Description

The dataset used is the "From Group to Individual Labels using Deep Features" (Kotzias et al., KDD 2015). It consists of 3,000 labeled sentences selected from reviews on three specific websites, ensuring a balanced distribution:

* **Sources:** Amazon (Products), IMDb (Movies), Yelp (Restaurants).
* **Total Instances:** 2,726 (after cleaning duplicates/corrupt rows).
* **Class Balance:** Balanced binary classification (0 for Negative, 1 for Positive).
* **Original Format:** `sentence \t score`

## 4\. Preprocessing Steps

To convert raw text into a machine-readable format, a comprehensive preprocessing script was developed using the `nltk` library. The pipeline included:

1. **Normalization:** Converting all text to lowercase.
2. **Noise Removal:** Removing punctuation and words containing numbers (regex).
3. **Tokenization:** Splitting sentences into individual words.
4. **POS Tagging & Lemmatization:** Using Part-of-Speech tags to inform the `WordNetLemmatizer`, reducing words to their root form (e.g., "better" $\to$ "good") to reduce sparsity.
5. **Stopword Removal:** Removing common English stopwords (e.g., "the", "is") that add little semantic value.
6. **Splitting:** The processed data was split into **80% Training (2,180 samples)** and **20% Testing (546 samples)** with stratification to maintain class balance.

## 5\. Models and Architecture

### 5.1 Baseline Model (Dummy Classifier)

A `DummyClassifier` with a 'stratified' strategy was used to establish a baseline. It generates predictions by respecting the class distribution of the training set, effectively guessing randomly based on class probability.

### 5.2 Dense Neural Network (DNN)

This model utilizes a "Bag of Words" approach via TF-IDF vectorization.

* **Input:** TF-IDF vectors (Vocabulary limit: 5,000).
* **Architecture:**
* Dense Layer (64 units, ReLU) + Dropout.
* Dense Layer (32 units, ReLU) + Dropout.
* Output Layer (1 unit, Sigmoid).
* **Optimization:** Tuned using GridSearchCV.

### 5.3 Vanilla RNN

A Recurrent Neural Network designed to capture sequential dependencies.

* **Input:** Tokenized and padded sequences (Length: 11, based on 90th percentile).
* **Architecture:**
* Embedding Layer (Dim: 50).
* SpatialDropout1D.
* SimpleRNN Layer (32 units, Dropout 0.4).
* Dense Output (Sigmoid).

### 5.4 LSTM (Long Short-Term Memory)

An advanced RNN designed to mitigate the vanishing gradient problem.

* **Input:** Tokenized and padded sequences (Length: 14, based on 95th percentile).
* **Architecture:**
* Embedding Layer (Dim: 50).
* SpatialDropout1D.
* LSTM Layer (64 units, Dropout 0.5).
* Dense Output (Sigmoid) with L2 Regularization.

## 6\. Training Process

Hyperparameter tuning was performed using `GridSearchCV` with 3-fold cross-validation.

* **DNN:** Best configuration found was `Optimizer: RMSprop`, `Dropout: 0.3`, `Batch Size: 32`.
* **RNN:** Trained for 20 epochs. It required a higher dropout (0.6) and smaller units (32) to combat overfitting inherent to simple RNNs.
* **LSTM:** Trained for 15 epochs. Best configuration used `Adam` optimizer and `64` LSTM units. Early stopping was employed to prevent overfitting.

## 7\. Comparative Analysis & Evaluation

The models were evaluated on the unseen test set (546 samples).

| Model | Accuracy | Precision | Recall | F1-Score | Cohen's Kappa |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 0.5220 | 0.5267 | 0.5018 | 0.5140 | 0.04 (Approx) |
| **Vanilla RNN** | 0.7418 | 0.7343 | 0.7636 | 0.7487 | 0.4833 |
| **LSTM** | 0.7875 | 0.7641 | **0.8364** | 0.7986 | 0.5748 |
| **Dense NN (TF-IDF)**| **0.8187** | **0.84** | 0.78 | **0.82** | **0.637** |

### Key Findings

1. **DNN Dominance:** The Dense Network with TF-IDF achieved the highest accuracy (**81.87%**). This suggests that for short sentences (reviews), the presence of specific keywords (e.g., "love", "bad") is more predictive than the sequence of words.
2. **LSTM vs. RNN:** The LSTM outperformed the Vanilla RNN (78.75% vs 74.18%) and achieved a higher Kappa score (0.57 vs 0.48), proving its superior ability to handle dependencies even in short texts.
3. **Overfitting:** The RNN and LSTM models showed signs of overfitting (Train accuracy \> 98% vs Test accuracy \~75-78%), despite aggressive dropout and regularization.

## 8\. Feature Importance

Feature analysis of the Dense Network revealed the most influential words:

* **Negative Predictors:** "Cheap", "Tired", "Poor", "Unbelievable" (in negative context).
* **Positive Predictors:** "Love", "Quality", "Service", "Great".

## 9\. Conclusions and Future Work

The project successfully implemented and compared three neural network architectures. The **Dense Neural Network with TF-IDF** proved to be the most efficient model for this specific dataset, likely due to the short length of the reviews where sequential context is less critical than keyword presence.

**Future Work:**

* **Pre-trained Embeddings:** utilizing GloVe or Word2Vec instead of learning embeddings from scratch to improve LSTM performance.
* **Transformer Models:** Implementing BERT or RoBERTa to capture deeper contextual nuances.
* **Data Augmentation:** Increasing the dataset size to reduce overfitting in sequential models.
