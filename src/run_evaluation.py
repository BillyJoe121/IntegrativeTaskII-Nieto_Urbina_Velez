"""
Run Evaluation for All Models

This script loads trained models and generates comprehensive evaluation reports
for Dense NN, RNN, and LSTM models. It organizes outputs according to the
project structure.

Usage:
    python src/run_evaluation.py --model dense
    python src/run_evaluation.py --model rnn
    python src/run_evaluation.py --model lstm
    python src/run_evaluation.py --all

Author: Nieto, Urbina, Velez
Course: Teoría de la Computación - Unidad 3
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import (
    calculate_performance_metrics,
    generate_model_report,
    plot_confusion_matrix,
    plot_training_history
)

# Import ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


# ============================================================================
# CONFIGURATION
# ============================================================================

# Get the directory where this script is located (src/)
SCRIPT_DIR = Path(__file__).resolve().parent

# Define paths relative to the script location
# PROJECT_ROOT is one level up from src/
PROJECT_ROOT = SCRIPT_DIR.parent 

DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'outputs' / 'saved_models'
METRICS_DIR = PROJECT_ROOT / 'outputs' / 'metrics'
FIGURES_DIR = PROJECT_ROOT / 'outputs' / 'figures'

# Ensure directories exist
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """
    Load preprocessed training and test data.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*70)
    print("  LOADING DATA")
    print("="*70 + "\n")
    
    train_path = DATA_DIR / 'train_data.csv'
    test_path = DATA_DIR / 'test_data.csv'
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Data files not found. Please run preprocessing first.\n"
            f"Expected: {train_path} and {test_path}"
        )
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df['processed_sentence'].fillna('')
    y_train = train_df['sentiment']
    X_test = test_df['processed_sentence'].fillna('')
    y_test = test_df['sentiment']
    
    print(f" Training samples: {len(X_train)}")
    print(f" Test samples: {len(X_test)}")
    print(f" Data loaded successfully!\n")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# DENSE NN EVALUATION
# ============================================================================

def evaluate_dense_nn(X_train, X_test, y_train, y_test):
    """
    Evaluate Dense Neural Network model.
    
    This function expects:
    - Trained model saved as: outputs/saved_models/dense_nn_model.h5
    - Vectorizer saved as: outputs/saved_models/tfidf_vectorizer.pkl
    - Training history saved as: outputs/saved_models/dense_nn_history.pkl
    """
    print("\n" + "="*70)
    print("  EVALUATING DENSE NEURAL NETWORK")
    print("="*70 + "\n")
    
    model_path = MODELS_DIR / 'dense_nn_model.keras'
    vectorizer_path = MODELS_DIR / 'tfidf_vectorizer.pkl'
    history_path = MODELS_DIR / 'dense_nn_history.pkl'
    
    # Check if files exist
    if not model_path.exists():
        print(f" Model not found: {model_path}")
        print(" Please train the Dense NN model first in the notebook.")
        return
    
    # Load model
    print(" Loading Dense NN model...")
    model = load_model(model_path)
    
    # Load vectorizer
    print(" Loading TF-IDF vectorizer...")
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load history (optional)
    history = None
    if history_path.exists():
        print(" Loading training history...")
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
    else:
        print(" Training history not found (will skip history plots)")
    
    # Transform data
    print(" Transforming data with TF-IDF...")
    X_train_tfidf = vectorizer.transform(X_train).toarray()
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    
    # Make predictions
    print(" Making predictions...")
    y_pred_probs = model.predict(X_test_tfidf, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Calculate metrics
    print(" Calculating metrics...")
    metrics = calculate_performance_metrics(y_test, y_pred, y_pred_probs)
    
    # Generate comprehensive report
    print("\n Generating comprehensive evaluation report...")
    generate_model_report(
        model=model,
        metrics=metrics,
        history=history,
        model_name="Dense Neural Network",
        model_type='dense',
        vectorizer=vectorizer,
        y_true=y_test,
        y_pred=y_pred,
        save_dir=str(METRICS_DIR)
    )
    
    # Generate visualizations
    print(" Generating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        "Dense Neural Network",
        save_path=str(FIGURES_DIR / 'dense_nn_confusion_matrix.png')
    )
    
    # Training history (if available)
    if history is not None:
        plot_training_history(
            history,
            "Dense Neural Network",
            save_path=str(FIGURES_DIR / 'dense_nn_training_history.png')
        )
    else:
        print(" Skipping training history plot (history not available)")

    
    print("\n Dense NN evaluation complete!")
    print(f" Reports saved to: {METRICS_DIR}")
    print(f" Figures saved to: {FIGURES_DIR}\n")


# ============================================================================
# RNN EVALUATION
# ============================================================================

def evaluate_rnn(X_train, X_test, y_train, y_test):
    """
    Evaluate vanilla RNN model.
    
    This function expects:
    - Trained model saved as: outputs/saved_models/rnn_model.h5
    - Tokenizer saved as: outputs/saved_models/rnn_tokenizer.pkl
    - Training history saved as: outputs/saved_models/rnn_history.pkl
    """
    print("\n" + "="*70)
    print("  EVALUATING VANILLA RNN")
    print("="*70 + "\n")
    
    model_path = MODELS_DIR / 'rnn_model.h5'
    tokenizer_path = MODELS_DIR / 'rnn_tokenizer.pkl'
    history_path = MODELS_DIR / 'rnn_history.pkl'
    
    # Check if files exist
    if not model_path.exists():
        print(f" Model not found: {model_path}")
        print(" Please train the RNN model first in the notebook.")
        print(" Add this code at the end of RNN section:")
        print("   rnn_model.save('../outputs/saved_models/rnn_model.h5')")
        print("   import pickle")
        print("   with open('../outputs/saved_models/rnn_tokenizer.pkl', 'wb') as f:")
        print("       pickle.dump(tokenizer_rnn, f)")
        print("   with open('../outputs/saved_models/rnn_history.pkl', 'wb') as f:")
        print("       pickle.dump(rnn_history.history, f)")
        return
    
    # Load model
    print(" Loading RNN model...")
    model = load_model(model_path)
    
    # Load tokenizer
    print(" Loading tokenizer...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load history
    print(" Loading training history...")
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    # Get max_length from model input shape
    max_length = model.input_shape[1]
    
    # Transform data
    print(f" Tokenizing and padding sequences (max_length={max_length})...")
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
    
    # Make predictions
    print(" Making predictions...")
    y_pred_probs = model.predict(X_test_padded, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Calculate metrics
    print(" Calculating metrics...")
    metrics = calculate_performance_metrics(y_test, y_pred, y_pred_probs)
    
    # Generate comprehensive report
    print("\n Generating comprehensive evaluation report...")
    generate_model_report(
        model=model,
        metrics=metrics,
        history=history,
        model_name="Vanilla RNN",
        model_type='rnn',
        vectorizer=None,
        y_true=y_test,
        y_pred=y_pred,
        save_dir=str(METRICS_DIR)
    )
    
    # Generate visualizations
    print(" Generating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        "Vanilla RNN",
        save_path=str(FIGURES_DIR / 'rnn_confusion_matrix.png')
    )
    
    # Training history
    plot_training_history(
        history,
        "Vanilla RNN",
        save_path=str(FIGURES_DIR / 'rnn_training_history.png')
    )
    
    print("\n RNN evaluation complete!")
    print(f" Reports saved to: {METRICS_DIR}")
    print(f" Figures saved to: {FIGURES_DIR}\n")


# ============================================================================
# LSTM EVALUATION
# ============================================================================

def evaluate_lstm(X_train, X_test, y_train, y_test):
    """
    Evaluate LSTM model.
    
    This function expects:
    - Trained model saved as: outputs/saved_models/lstm_model.h5
    - Tokenizer saved as: outputs/saved_models/lstm_tokenizer.pkl
    - Training history saved as: outputs/saved_models/lstm_history.pkl
    """
    print("\n" + "="*70)
    print("  EVALUATING LSTM")
    print("="*70 + "\n")
    
    model_path = MODELS_DIR / 'lstm_model.keras'
    tokenizer_path = MODELS_DIR / 'lstm_tokenizer.pkl'
    params_path = MODELS_DIR / 'lstm_params.pkl'
    history_path = MODELS_DIR / 'lstm_history.pkl'
    
    # Check if files exist
    if not model_path.exists():
        print(f" Model not found: {model_path}")
        print(" Please train the LSTM model first in the notebook.")
        return
    
    # Load model
    print(" Loading LSTM model...")
    model = load_model(model_path)
    
    # Load tokenizer
    print(" Loading tokenizer...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load parameters
    print(" Loading model parameters...")
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    max_length = params['max_length']
    
    # Load history (optional)
    history = None
    if history_path.exists():
        print(" Loading training history...")
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
    else:
        print(" Training history not found (will skip history plots)")
    
    # Transform data
    print(f" Tokenizing and padding sequences (max_length={max_length})...")
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
    
    # Make predictions
    print(" Making predictions...")
    y_pred_probs = model.predict(X_test_padded, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Calculate metrics
    print(" Calculating metrics...")
    metrics = calculate_performance_metrics(y_test, y_pred, y_pred_probs)
    
    # Generate comprehensive report
    print("\n Generating comprehensive evaluation report...")
    generate_model_report(
        model=model,
        metrics=metrics,
        history=history,
        model_name="LSTM",
        model_type='lstm',
        vectorizer=None,
        y_true=y_test,
        y_pred=y_pred,
        save_dir=str(METRICS_DIR)
    )
    
    # Generate visualizations
    print(" Generating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        "LSTM",
        save_path=str(FIGURES_DIR / 'lstm_confusion_matrix.png')
    )
    
    # Training history (if available)
    if history is not None:
        plot_training_history(
            history,
            "LSTM",
            save_path=str(FIGURES_DIR / 'lstm_training_history.png')
        )
    else:
        print(" Skipping training history plot (history not available)")

    
    print("\n LSTM evaluation complete!")
    print(f" Reports saved to: {METRICS_DIR}")
    print(f" Figures saved to: {FIGURES_DIR}\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run evaluation for sentiment analysis models'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['dense', 'rnn', 'lstm', 'all'],
        default='all',
        help='Which model to evaluate (default: all)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  SENTIMENT ANALYSIS MODEL EVALUATION")
    print("  Nieto, Urbina, Velez")
    print("="*70)
    
    # Load data
    try:
        X_train, X_test, y_train, y_test = load_data()
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        return
    
    # Run evaluations based on argument
    if args.model == 'dense' or args.model == 'all':
        evaluate_dense_nn(X_train, X_test, y_train, y_test)
    
    if args.model == 'rnn' or args.model == 'all':
        if args.model == 'rnn':
            print("\n" + "="*70)
            print("  WARNING: RNN model not available")
            print("="*70)
            print("\n RNN model was not trained in the notebook.")
            print(" Only Dense NN and LSTM models are available.\n")
        # Skip RNN evaluation silently when running --all
    
    if args.model == 'lstm' or args.model == 'all':
        evaluate_lstm(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*70)
    print("  EVALUATION COMPLETE")
    print("="*70)
    print(f"\n Check outputs:")
    print(f"   Reports: {METRICS_DIR}")
    print(f"   Figures: {FIGURES_DIR}\n")


if __name__ == "__main__":
    main()
