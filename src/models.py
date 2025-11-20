"""
Model Architectures for Sentiment Analysis

This module contains the neural network architectures used for sentiment
classification: Dense NN, Vanilla RNN, and LSTM.

These functions can be imported by notebooks and scripts to ensure consistency
in model definitions across the project.

Author: Nieto, Urbina, Velez
Course: Teoría de la Computación - Unidad 3
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Embedding, LSTM, SimpleRNN, SpatialDropout1D
)
import tensorflow as tf


# ============================================================================
# DENSE NEURAL NETWORK
# ============================================================================

def create_dnn_model(input_dim, hidden_units=64, dropout_rate=0.5, optimizer='adam'):
    """
    Creates a Dense Neural Network for sentiment classification.
    
    Architecture:
    - Input layer with ReLU activation
    - Hidden layer (32 units) with ReLU activation
    - Dropout layers for regularization
    - Output layer with sigmoid activation
    
    Parameters:
    -----------
    input_dim : int
        Dimension of input features (vocabulary size for TF-IDF)
    hidden_units : int
        Number of units in first hidden layer (default: 64)
    dropout_rate : float
        Dropout rate for regularization (default: 0.5)
    optimizer : str
        Optimizer to use (default: 'adam')
    
    Returns:
    --------
    model : Sequential
        Compiled Dense NN model
    """
    model = Sequential([
        # Input Layer
        Dense(hidden_units, activation='relu', input_dim=input_dim, name='input_layer'),
        Dropout(dropout_rate, name='dropout_1'),
        
        # Hidden Layer
        Dense(32, activation='relu', name='hidden_layer'),
        Dropout(dropout_rate, name='dropout_2'),
        
        # Output Layer
        Dense(1, activation='sigmoid', name='output_layer')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# LSTM MODEL
# ============================================================================

def create_lstm_model(vocab_size, max_length, embedding_dim=50, lstm_units=64, 
                     dropout_rate=0.5, optimizer='adam'):
    """
    Creates an LSTM model for sentiment classification.
    
    ANTI-OVERFITTING MEASURES:
    - Small embedding dimension (50)
    - Single LSTM layer (avoids deep architecture overfitting)
    - Aggressive dropout (0.5 default)
    - SpatialDropout1D for embeddings
    - L2 regularization on Dense layer
    
    Architecture:
    - Embedding layer (learns word representations)
    - SpatialDropout1D (40% of dropout_rate)
    - Single LSTM layer with recurrent dropout
    - Dropout layer
    - Dense output layer with L2 regularization
    
    Parameters:
    -----------
    vocab_size : int
        Size of the vocabulary
    max_length : int
        Maximum sequence length
    embedding_dim : int
        Dimension of word embeddings (default: 50)
    lstm_units : int
        Number of LSTM units (default: 64)
    dropout_rate : float
        Dropout rate (default: 0.5)
    optimizer : str
        Optimizer to use (default: 'adam')
    
    Returns:
    --------
    model : Sequential
        Compiled LSTM model
    """
    model = Sequential([
        # Embedding layer - learns word representations
        Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim, 
            input_length=max_length,
            name='embedding_layer'
        ),
        
        # Spatial dropout for embeddings (more effective than regular dropout)
        SpatialDropout1D(dropout_rate * 0.4, name='spatial_dropout'),
        
        # Single LSTM layer - AVOID stacking multiple LSTMs (causes overfitting)
        LSTM(
            units=lstm_units, 
            dropout=dropout_rate * 0.4,              # Recurrent dropout
            recurrent_dropout=dropout_rate * 0.3,    # Additional recurrent dropout
            name='lstm_layer'
        ),
        
        # Regular dropout before dense layer
        Dropout(dropout_rate, name='dropout'),
        
        # Output layer with L2 regularization
        Dense(
            1, 
            activation='sigmoid', 
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name='output_layer'
        )
    ])
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# VANILLA RNN MODEL
# ============================================================================

def create_rnn_model(vocab_size, max_length, embedding_dim=50, rnn_units=48, 
                    dropout_rate=0.5, optimizer='adam'):
    """
    Creates a Vanilla RNN model for sentiment classification.
    
    ANTI-OVERFITTING MEASURES:
    - Small embedding dimension (50)
    - Single SimpleRNN layer with moderate units (48 default)
    - Aggressive dropout (0.5 default)
    - SpatialDropout1D for embeddings
    - L2 regularization on Dense layer
    - Smaller capacity than LSTM (RNNs overfit more easily)
    
    DESIGN RATIONALE:
    - Vanilla RNN lacks memory cells, so it needs less capacity
    - More prone to vanishing gradients, so keep architecture simple
    - Higher dropout compensates for lack of gating mechanisms
    
    Architecture:
    - Embedding layer (learns word representations)
    - SpatialDropout1D (50% of dropout_rate)
    - Single SimpleRNN layer with recurrent dropout
    - Dropout layer
    - Dense output layer with L2 regularization
    
    Parameters:
    -----------
    vocab_size : int
        Size of the vocabulary
    max_length : int
        Maximum sequence length
    embedding_dim : int
        Dimension of word embeddings (default: 50)
    rnn_units : int
        Number of RNN units (default: 48, smaller than LSTM)
    dropout_rate : float
        Dropout rate (default: 0.5, higher than LSTM)
    optimizer : str
        Optimizer to use (default: 'adam')
    
    Returns:
    --------
    model : Sequential
        Compiled Vanilla RNN model
    """
    model = Sequential([
        # Embedding layer - learns word representations
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            name='embedding_layer'
        ),
        
        # Spatial dropout for embeddings (more effective than regular dropout)
        SpatialDropout1D(dropout_rate * 0.5, name='spatial_dropout'),
        
        # Single SimpleRNN layer - AVOID stacking (causes severe overfitting)
        SimpleRNN(
            units=rnn_units,
            dropout=dropout_rate * 0.4,           # Input dropout
            recurrent_dropout=dropout_rate * 0.3, # Recurrent dropout
            return_sequences=False,                # Only return final output
            name='rnn_layer'
        ),
        
        # Regular dropout before dense layer (aggressive)
        Dropout(dropout_rate, name='dropout'),
        
        # Output layer with L2 regularization
        Dense(
            1, 
            activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name='output_layer'
        )
    ])
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_summary(model_type='all'):
    """
    Print summary information about available models.
    
    Parameters:
    -----------
    model_type : str
        Type of model to summarize ('dense', 'lstm', 'rnn', or 'all')
    """
    summaries = {
        'dense': {
            'name': 'Dense Neural Network',
            'input': 'TF-IDF vectors',
            'architecture': 'Input(64) -> Hidden(32) -> Output(1)',
            'parameters': '~5K (depends on vocabulary)',
            'best_for': 'Fast inference, bag-of-words tasks'
        },
        'lstm': {
            'name': 'LSTM (Long Short-Term Memory)',
            'input': 'Tokenized sequences',
            'architecture': 'Embedding(50) -> LSTM(64) -> Output(1)',
            'parameters': '~20K (depends on vocabulary)',
            'best_for': 'Long-term dependencies, sequential patterns'
        },
        'rnn': {
            'name': 'Vanilla RNN (SimpleRNN)',
            'input': 'Tokenized sequences',
            'architecture': 'Embedding(50) -> SimpleRNN(48) -> Output(1)',
            'parameters': '~15K (depends on vocabulary)',
            'best_for': 'Short sequences, simple patterns'
        }
    }
    
    if model_type == 'all':
        for key, info in summaries.items():
            print(f"\n{info['name']}:")
            print(f"  Input: {info['input']}")
            print(f"  Architecture: {info['architecture']}")
            print(f"  Parameters: {info['parameters']}")
            print(f"  Best for: {info['best_for']}")
    else:
        info = summaries.get(model_type.lower())
        if info:
            print(f"\n{info['name']}:")
            print(f"  Input: {info['input']}")
            print(f"  Architecture: {info['architecture']}")
            print(f"  Parameters: {info['parameters']}")
            print(f"  Best for: {info['best_for']}")
        else:
            print(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    print("Model Architectures Module")
    print("="*50)
    print("\nAvailable models:")
    print("  - create_dnn_model()")
    print("  - create_lstm_model()")
    print("  - create_rnn_model()")
    print("\nUse get_model_summary() for more information.")
