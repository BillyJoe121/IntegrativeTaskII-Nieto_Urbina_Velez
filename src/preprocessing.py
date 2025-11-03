import pandas as pd
import nltk
import re
import string
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag

## NLTK Resource Configuration

def download_nltk_resources():
    """Downloads necessary NLTK packages for the script."""
    print("Downloading required NLTK resources...")
    try:
        # Download necessary data for tokenization, stopwords, 
        # part-of-speech tagging, and lemmatization
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("NLTK resources are ready.")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")

## Text Preprocessing Functions

# Initialize lemmatizer and stopwords outside the function for efficiency
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(tag):
    """Maps NLTK POS tags to tags compatible with WordNet Lemmatizer."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def preprocess_text(text):
    """Applies the full text cleaning pipeline."""
    if not isinstance(text, str):
        return "" # Return an empty string if the input is not a string
        
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    # Remove words with numbers
    text = re.sub(r'\w*\d\w*', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # POS tagging
    pos_tags = pos_tag(tokens)
    # Lemmatization and stopword removal
    lemmatized_tokens = []
    for word, tag in pos_tags:
        #Remove stopwords and short words (len > 1)
        if word not in stop_words and len(word) > 1:
            # Get the correct POS tag
            pos = get_wordnet_pos(tag)
            # Lemmatize
            lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=pos))
    # Join tokens back into a string
    return " ".join(lemmatized_tokens)

## Main Script Function

def main():
    """Loads, processes, splits, and saves the data."""
    
    # Define paths
    # This script assumes it is run from the project's root directory
    input_path = 'data/processed/cleaned_data.csv'
    train_output_path = 'data/processed/train_data.csv'
    test_output_path = 'data/processed/test_data.csv'
    
    #Load the cleaned dataset (from EDA)
    print(f"Loading cleaned data from {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        print("Please run the '01_eda.ipynb' notebook first to generate this file.")
        return

    df = pd.read_csv(input_path)
    
    # Apply text preprocessing
    print("Applying text preprocessing (lemmatization, etc.)...")
    # Ensure 'sentence' column is treated as string
    df['processed_sentence'] = df['sentence'].astype(str).apply(preprocess_text)
    print("Text preprocessing complete.")
    
    # Define X (features) and y (labels)
    # We will use the newly 'processed_sentence' for the models
    X = df['processed_sentence']
    y = df['sentiment']

    # Split into training and test sets (80/20)
    print("Splitting data into training and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42, # Ensures reproducible results
        stratify=y       # Maintains class balance in both sets
    )

    print(f"Training data points: {len(X_train)}")
    print(f"Test data points: {len(X_test)}")

    # Save the split datasets to 'data/processed'
    print(f"Saving training data to {train_output_path}...")
    # Combine X_train and y_train into a single DataFrame for saving
    train_df = pd.DataFrame({'processed_sentence': X_train, 'sentiment': y_train})
    train_df.to_csv(train_output_path, index=False)

    print(f"Saving test data to {test_output_path}...")
    # Combine X_test and y_test into a single DataFrame
    test_df = pd.DataFrame({'processed_sentence': X_test, 'sentiment': y_test})
    test_df.to_csv(test_output_path, index=False)
    
    print("\nPreprocessing and data splitting complete!")

## Script Execution
if __name__ == "__main__":
    # This block ensures the code runs only when the script is executed directly
    download_nltk_resources()
    main()