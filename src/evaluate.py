"""
Evaluation Module for Sentiment Analysis Models

This module provides comprehensive evaluation capabilities for neural network models,
including performance metrics, training complexity analysis, interpretability analysis,
and connections to Turing Machine theoretical concepts.

Author: Nieto, Urbina, Velez
Course: Teoría de la Computación - Unidad 3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score
)
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# OUTPUT DIRECTORY STRUCTURE
# ============================================================================
# This module organizes outputs according to the project structure:
#   outputs/
#   ├── figures/          - plots, word clouds, charts used in report
#   ├── metrics/          - model performance metrics or confusion matrices
#   └── saved_models/     - trained model weights (if small enough)
# ============================================================================


# ============================================================================
# PERFORMANCE METRICS ANALYSIS
# ============================================================================

def calculate_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary containing all performance metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
    }
    
    # Calculate confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def print_performance_report(
    metrics: Dict[str, float],
    model_name: str,
    baseline_accuracy: float = 0.5220
) -> None:
    """
    Print a formatted performance report.
    
    Args:
        metrics: Dictionary of performance metrics
        model_name: Name of the model
        baseline_accuracy: Baseline accuracy for comparison
    """
    print(f"\n{'='*70}")
    print(f"  PERFORMANCE EVALUATION: {model_name}")
    print(f"{'='*70}\n")
    
    print(f" Classification Metrics:")
    print(f"  • Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  • Precision:      {metrics['precision']:.4f}")
    print(f"  • Recall:         {metrics['recall']:.4f}")
    print(f"  • F1-Score:       {metrics['f1_score']:.4f}")
    print(f"  • Cohen's Kappa:  {metrics['cohen_kappa']:.4f}")
    
    if 'specificity' in metrics:
        print(f"  • Specificity:    {metrics['specificity']:.4f}")
    
    print(f"\n Confusion Matrix Components:")
    if 'true_positives' in metrics:
        print(f"  • True Positives:  {metrics['true_positives']}")
        print(f"  • True Negatives:  {metrics['true_negatives']}")
        print(f"  • False Positives: {metrics['false_positives']}")
        print(f"  • False Negatives: {metrics['false_negatives']}")
    
    # Comparison with baseline
    improvement = (metrics['accuracy'] - baseline_accuracy) * 100
    print(f"\n Comparison with Baseline:")
    print(f"  • Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"  • Model Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  • Improvement:       {improvement:+.2f}%")
    
    if improvement > 0:
        print(f"   Model outperforms baseline by {improvement:.2f}%")
    else:
        print(f"    Model underperforms baseline by {abs(improvement):.2f}%")
    
    print(f"\n{'='*70}\n")


# ============================================================================
# TRAINING COMPLEXITY ANALYSIS
# ============================================================================

def analyze_model_complexity(model: Any) -> Dict[str, Any]:
    """
    Analyze the complexity of a neural network model.
    
    Args:
        model: Keras/TensorFlow model
    
    Returns:
        Dictionary containing complexity metrics
    """
    complexity = {}
    
    try:
        # Get parameter counts
        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
        non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        
        complexity['total_parameters'] = int(total_params)
        complexity['trainable_parameters'] = int(trainable_params)
        complexity['non_trainable_parameters'] = int(non_trainable_params)
        
        # Estimate memory footprint (rough estimate in MB)
        # Assuming float32 (4 bytes per parameter)
        memory_mb = (total_params * 4) / (1024 * 1024)
        complexity['estimated_memory_mb'] = round(memory_mb, 2)
        
        # Layer information
        complexity['num_layers'] = len(model.layers)
        complexity['layer_types'] = [layer.__class__.__name__ for layer in model.layers]
        
        # Layer-wise parameters
        layer_params = []
        for layer in model.layers:
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'parameters': int(layer.count_params())
            }
            layer_params.append(layer_info)
        complexity['layer_parameters'] = layer_params
        
    except Exception as e:
        complexity['error'] = str(e)
    
    return complexity


def analyze_training_efficiency(history: Any) -> Dict[str, Any]:
    """
    Analyze training efficiency from training history.
    
    Args:
        history: Keras History object or dict with 'loss', 'accuracy', etc.
    
    Returns:
        Dictionary containing training efficiency metrics
    """
    efficiency = {}
    
    try:
        # Extract history data
        if hasattr(history, 'history'):
            hist_dict = history.history
        else:
            hist_dict = history
        
        # Number of epochs trained
        epochs_trained = len(hist_dict.get('loss', []))
        efficiency['epochs_trained'] = epochs_trained
        
        # Final metrics
        if 'loss' in hist_dict:
            efficiency['final_train_loss'] = float(hist_dict['loss'][-1])
        if 'accuracy' in hist_dict:
            efficiency['final_train_accuracy'] = float(hist_dict['accuracy'][-1])
        if 'val_loss' in hist_dict:
            efficiency['final_val_loss'] = float(hist_dict['val_loss'][-1])
        if 'val_accuracy' in hist_dict:
            efficiency['final_val_accuracy'] = float(hist_dict['val_accuracy'][-1])
        
        # Convergence analysis
        if 'val_loss' in hist_dict and len(hist_dict['val_loss']) > 1:
            val_losses = hist_dict['val_loss']
            best_epoch = int(np.argmin(val_losses)) + 1
            efficiency['best_epoch'] = best_epoch
            efficiency['best_val_loss'] = float(min(val_losses))
            
            # Check for early stopping
            if best_epoch < epochs_trained:
                efficiency['early_stopped'] = True
                efficiency['epochs_after_best'] = epochs_trained - best_epoch
            else:
                efficiency['early_stopped'] = False
        
        # Overfitting detection
        if 'loss' in hist_dict and 'val_loss' in hist_dict:
            final_train_loss = hist_dict['loss'][-1]
            final_val_loss = hist_dict['val_loss'][-1]
            gap = final_val_loss - final_train_loss
            efficiency['train_val_gap'] = float(gap)
            
            if gap > 0.1:
                efficiency['overfitting_detected'] = True
                efficiency['overfitting_severity'] = 'High' if gap > 0.3 else 'Moderate'
            else:
                efficiency['overfitting_detected'] = False
        
    except Exception as e:
        efficiency['error'] = str(e)
    
    return efficiency


def print_complexity_report(
    complexity: Dict[str, Any],
    efficiency: Dict[str, Any],
    model_name: str
) -> None:
    """
    Print a formatted complexity and efficiency report.
    """
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLEXITY ANALYSIS: {model_name}")
    print(f"{'='*70}\n")
    
    print(f"  Model Architecture:")
    print(f"  • Total Parameters:        {complexity.get('total_parameters', 'N/A'):,}")
    print(f"  • Trainable Parameters:    {complexity.get('trainable_parameters', 'N/A'):,}")
    print(f"  • Non-trainable Parameters: {complexity.get('non_trainable_parameters', 'N/A'):,}")
    print(f"  • Estimated Memory:        {complexity.get('estimated_memory_mb', 'N/A')} MB")
    print(f"  • Number of Layers:        {complexity.get('num_layers', 'N/A')}")
    
    print(f"\n Layer-wise Parameters:")
    if 'layer_parameters' in complexity:
        for layer_info in complexity['layer_parameters']:
            print(f"  • {layer_info['name']:20s} ({layer_info['type']:15s}): {layer_info['parameters']:>8,} params")
    
    # --- CORRECCIÓN AQUÍ: Función auxiliar para formatear ---
    def safe_fmt(val):
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return str(val)
    
    print(f"\n Training Efficiency:")
    print(f"  • Epochs Trained:          {efficiency.get('epochs_trained', 'N/A')}")
    if 'best_epoch' in efficiency:
        print(f"  • Best Epoch:              {efficiency['best_epoch']}")
        
    # Usamos safe_fmt en lugar de poner :.4f directamente
    print(f"  • Final Train Loss:        {safe_fmt(efficiency.get('final_train_loss', 'N/A'))}")
    print(f"  • Final Train Accuracy:    {safe_fmt(efficiency.get('final_train_accuracy', 'N/A'))}")
    
    if 'final_val_loss' in efficiency:
        print(f"  • Final Val Loss:          {safe_fmt(efficiency['final_val_loss'])}")
        print(f"  • Final Val Accuracy:      {safe_fmt(efficiency['final_val_accuracy'])}")
    
    if 'overfitting_detected' in efficiency:
        print(f"\n Overfitting Analysis:")
        print(f"  • Train-Val Gap:           {safe_fmt(efficiency.get('train_val_gap', 0))}")
        if efficiency['overfitting_detected']:
            severity = efficiency.get('overfitting_severity', 'Unknown')
            print(f"    Overfitting Detected:   {severity}")
        else:
            print(f"   No Significant Overfitting")
    
    print(f"\n{'='*70}\n")
    
# ============================================================================
# INTERPRETABILITY ANALYSIS (Dense NN Specific)
# ============================================================================

def analyze_dense_weights(
    model: Any,
    vectorizer: Any,
    top_n: int = 20
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Analyze weights of the first Dense layer to identify important features.
    
    Args:
        model: Trained Keras model
        vectorizer: TF-IDF vectorizer used for feature extraction
        top_n: Number of top features to return
    
    Returns:
        Dictionary with top positive and negative features
    """
    try:
        # Get the first Dense layer weights
        first_layer_weights = model.layers[0].get_weights()[0]  # Shape: (vocab_size, hidden_units)
        
        # Average weights across all hidden units
        avg_weights = np.mean(first_layer_weights, axis=1)
        
        # Get feature names from vectorizer
        feature_names = vectorizer.get_feature_names_out()
        
        # Create feature-weight pairs
        feature_weights = list(zip(feature_names, avg_weights))
        
        # Sort by weight
        sorted_features = sorted(feature_weights, key=lambda x: x[1], reverse=True)
        
        # Get top positive and negative features
        top_positive = sorted_features[:top_n]
        top_negative = sorted_features[-top_n:][::-1]  # Reverse to show most negative first
        
        return {
            'top_positive_features': top_positive,
            'top_negative_features': top_negative,
            'all_features': sorted_features
        }
    
    except Exception as e:
        print(f"Error analyzing weights: {e}")
        return {}


def visualize_feature_importance(
    feature_weights: Dict[str, List[Tuple[str, float]]],
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize feature importance with horizontal bar charts.
    
    Args:
        feature_weights: Dictionary from analyze_dense_weights
        model_name: Name of the model
        save_path: Path to save the figure (optional)
    """
    if not feature_weights:
        print("No feature weights to visualize.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Positive features
    pos_features = feature_weights['top_positive_features']
    pos_words = [f[0] for f in pos_features]
    pos_weights = [f[1] for f in pos_features]
    
    ax1.barh(range(len(pos_words)), pos_weights, color='green', alpha=0.7)
    ax1.set_yticks(range(len(pos_words)))
    ax1.set_yticklabels(pos_words)
    ax1.set_xlabel('Average Weight', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_name}\nTop Positive Sentiment Features', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Negative features
    neg_features = feature_weights['top_negative_features']
    neg_words = [f[0] for f in neg_features]
    neg_weights = [f[1] for f in neg_features]
    
    ax2.barh(range(len(neg_words)), neg_weights, color='red', alpha=0.7)
    ax2.set_yticks(range(len(neg_words)))
    ax2.set_yticklabels(neg_words)
    ax2.set_xlabel('Average Weight', fontsize=12, fontweight='bold')
    ax2.set_title(f'{model_name}\nTop Negative Sentiment Features', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Feature importance plot saved to: {save_path}")
    
    plt.show()


def print_interpretability_report(
    feature_weights: Dict[str, List[Tuple[str, float]]],
    model_name: str,
    top_n: int = 10
) -> None:
    """
    Print interpretability analysis report.
    
    Args:
        feature_weights: Dictionary from analyze_dense_weights
        model_name: Name of the model
        top_n: Number of features to display
    """
    print(f"\n{'='*70}")
    print(f"  INTERPRETABILITY ANALYSIS: {model_name}")
    print(f"{'='*70}\n")
    
    print(f" Top {top_n} Features for POSITIVE Sentiment:")
    pos_features = feature_weights['top_positive_features'][:top_n]
    for i, (word, weight) in enumerate(pos_features, 1):
        print(f"  {i:2d}. {word:20s} → {weight:+.4f}")
    
    print(f"\n Top {top_n} Features for NEGATIVE Sentiment:")
    neg_features = feature_weights['top_negative_features'][:top_n]
    for i, (word, weight) in enumerate(neg_features, 1):
        print(f"  {i:2d}. {word:20s} → {weight:+.4f}")
    
    print(f"\n{'='*70}\n")


# ============================================================================
# TURING MACHINE CONCEPTUAL ANALYSIS
# ============================================================================

def analyze_turing_concepts(
    model_type: str,
    architecture_info: Dict[str, Any]
) -> str:
    """
    Analyze the model in terms of Turing Machine concepts.
    
    Args:
        model_type: Type of model ('dense', 'rnn', 'lstm')
        architecture_info: Dictionary with architecture details
    
    Returns:
        Markdown-formatted analysis string
    """
    analysis = f"\n{'='*70}\n"
    analysis += f"  TURING MACHINE CONCEPTUAL ANALYSIS: {model_type.upper()}\n"
    analysis += f"{'='*70}\n\n"
    
    if model_type.lower() == 'dense':
        analysis += " **MEMORY CHARACTERISTICS**\n"
        analysis += "  • Type: STATELESS (No memory between inputs)\n"
        analysis += "  • Comparison to Turing Machine:\n"
        analysis += "    - A Turing Machine has a TAPE for memory storage\n"
        analysis += "    - Dense NN has NO sequential memory - each input is independent\n"
        analysis += "    - Similar to a finite automaton without state transitions\n"
        analysis += "  • Implications:\n"
        analysis += "    - Cannot capture temporal dependencies\n"
        analysis += "    - Treats each sentence as a bag-of-words (order-independent)\n"
        analysis += "    - Suitable for tasks where word order is less critical\n\n"
        
        analysis += " **SEQUENCE PROCESSING**\n"
        analysis += "  • Processing Model: PARALLEL (all features processed simultaneously)\n"
        analysis += "  • Comparison to Turing Machine:\n"
        analysis += "    - Turing Machine processes symbols SEQUENTIALLY on tape\n"
        analysis += "    - Dense NN processes entire TF-IDF vector at once\n"
        analysis += "    - No concept of 'reading head' moving across input\n"
        analysis += "  • Implications:\n"
        analysis += "    - Fast inference (no sequential bottleneck)\n"
        analysis += "    - Cannot model long-range dependencies\n"
        analysis += "    - Word order information is lost\n\n"
        
        analysis += "  **COMPUTABILITY & EXPRESSIVENESS**\n"
        analysis += "  • Computational Power:\n"
        analysis += "    - Dense NNs are UNIVERSAL FUNCTION APPROXIMATORS\n"
        analysis += "    - Can approximate any continuous function (Universal Approximation Theorem)\n"
        analysis += "    - However, NOT Turing-complete (cannot simulate arbitrary computation)\n"
        analysis += "  • Limitations:\n"
        analysis += "    - Fixed input size (vocabulary dimension)\n"
        analysis += "    - Cannot handle variable-length sequences naturally\n"
        analysis += "    - No internal state or memory mechanism\n"
        analysis += "  • Comparison to Turing Machine:\n"
        analysis += "    - Turing Machine: Can compute any computable function\n"
        analysis += "    - Dense NN: Limited to function approximation over fixed inputs\n\n"
        
        analysis += " **CONNECTION TO FORMAL LANGUAGE THEORY**\n"
        analysis += "  • Language Recognition Capability:\n"
        analysis += "    - Can recognize REGULAR LANGUAGES (like finite automata)\n"
        analysis += "    - Cannot recognize CONTEXT-FREE or CONTEXT-SENSITIVE languages\n"
        analysis += "    - Sentiment classification is a REGULAR task (word presence/absence)\n"
        analysis += "  • Chomsky Hierarchy Position:\n"
        analysis += "    - Level 3: Regular Languages \n"
        analysis += "    - Level 2: Context-Free Languages ❌\n"
        analysis += "    - Level 1: Context-Sensitive Languages ❌\n"
        analysis += "    - Level 0: Recursively Enumerable Languages ❌\n\n"
    
    elif model_type.lower() == 'rnn':
        analysis += " **MEMORY CHARACTERISTICS**\n"
        analysis += "  • Type: STATEFUL (Hidden state carries information)\n"
        analysis += "  • Comparison to Turing Machine:\n"
        analysis += "    - RNN has HIDDEN STATE (analogous to TM's internal state)\n"
        analysis += "    - Limited memory capacity (vanishing gradient problem)\n"
        analysis += "    - Cannot access arbitrary past positions (unlike TM tape)\n"
        analysis += "  • Implications:\n"
        analysis += "    - Can capture SHORT-TERM dependencies\n"
        analysis += "    - Struggles with long sequences\n"
        analysis += "    - Memory decays over time\n\n"
        
        analysis += " **SEQUENCE PROCESSING**\n"
        analysis += "  • Processing Model: SEQUENTIAL (left-to-right)\n"
        analysis += "  • Comparison to Turing Machine:\n"
        analysis += "    - Similar to TM: processes one token at a time\n"
        analysis += "    - Updates internal state at each step\n"
        analysis += "    - Cannot move backwards (unlike TM's bidirectional tape)\n"
        analysis += "  • Implications:\n"
        analysis += "    - Captures word order and context\n"
        analysis += "    - Slower inference than Dense NN\n"
        analysis += "    - Better for sequential tasks\n\n"
        
        analysis += "  **COMPUTABILITY & EXPRESSIVENESS**\n"
        analysis += "  • Computational Power:\n"
        analysis += "    - RNNs are TURING-COMPLETE (theoretically)\n"
        analysis += "    - Can simulate any Turing Machine computation\n"
        analysis += "    - In practice: limited by vanishing gradients\n"
        analysis += "  • Limitations:\n"
        analysis += "    - Finite precision (vs TM's infinite tape)\n"
        analysis += "    - Gradient-based training constraints\n"
        analysis += "    - Difficulty learning long-term dependencies\n\n"
    
    elif model_type.lower() == 'lstm':
        analysis += " **MEMORY CHARACTERISTICS**\n"
        analysis += "  • Type: STATEFUL with LONG-TERM MEMORY (Cell state + Hidden state)\n"
        analysis += "  • Comparison to Turing Machine:\n"
        analysis += "    - LSTM has CELL STATE (analogous to TM's tape)\n"
        analysis += "    - GATES control information flow (like TM's transition function)\n"
        analysis += "    - Can selectively remember/forget information\n"
        analysis += "  • Implications:\n"
        analysis += "    - Handles LONG-TERM dependencies effectively\n"
        analysis += "    - Mitigates vanishing gradient problem\n"
        analysis += "    - More sophisticated memory management than RNN\n\n"
        
        analysis += " **SEQUENCE PROCESSING**\n"
        analysis += "  • Processing Model: SEQUENTIAL with GATED MEMORY\n"
        analysis += "  • Comparison to Turing Machine:\n"
        analysis += "    - Similar sequential processing to TM\n"
        analysis += "    - Forget gate: decides what to remove from memory\n"
        analysis += "    - Input gate: decides what new information to store\n"
        analysis += "    - Output gate: decides what to output\n"
        analysis += "  • Implications:\n"
        analysis += "    - Excellent for long sequences\n"
        analysis += "    - Can learn complex temporal patterns\n"
        analysis += "    - More parameters = more computational cost\n\n"
        
        analysis += "  **COMPUTABILITY & EXPRESSIVENESS**\n"
        analysis += "  • Computational Power:\n"
        analysis += "    - LSTMs are TURING-COMPLETE (theoretically)\n"
        analysis += "    - More practical than vanilla RNNs for complex tasks\n"
        analysis += "    - Can approximate any sequential function\n"
        analysis += "  • Advantages over RNN:\n"
        analysis += "    - Better gradient flow (no vanishing gradient)\n"
        analysis += "    - Can learn dependencies across 100+ time steps\n"
        analysis += "    - More stable training\n\n"
    
    analysis += f"{'='*70}\n"
    return analysis


# ============================================================================
# COMPREHENSIVE REPORT GENERATION
# ============================================================================

def generate_model_report(
    model: Any,
    metrics: Dict[str, float],
    history: Any,
    model_name: str,
    model_type: str = 'dense',
    vectorizer: Any = None,
    y_true: np.ndarray = None,
    y_pred: np.ndarray = None,
    save_dir: str = '../outputs/metrics'  # Metrics and reports go here
) -> str:
    """
    Generate a comprehensive evaluation report for a model.
    
    Output files will be saved to:
        - Text reports: {save_dir}/*.txt (outputs/metrics/)
        - JSON metrics: {save_dir}/*.json (outputs/metrics/)
        - Visualizations: {save_dir}/../figures/*.png (outputs/figures/)
    
    Args:
        model: Trained model
        metrics: Performance metrics dictionary
        history: Training history
        model_name: Name of the model
        model_type: Type of model ('dense', 'rnn', 'lstm')
        vectorizer: Feature vectorizer (for Dense NN)
        y_true: True labels (for confusion matrix)
        y_pred: Predicted labels (for confusion matrix)
        save_dir: Directory to save metrics and reports (default: '../outputs/metrics')
    
    Returns:
        Formatted report string
    """
    report = []
    report.append(f"\n{'#'*70}")
    report.append(f"# COMPREHENSIVE EVALUATION REPORT: {model_name}")
    report.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"{'#'*70}\n")
    
    # 1. Performance Metrics
    report.append("\n## 1. PERFORMANCE METRICS\n")
    print_performance_report(metrics, model_name)
    
    # 2. Training Complexity
    complexity = analyze_model_complexity(model)
    efficiency = analyze_training_efficiency(history)
    report.append("\n## 2. TRAINING COMPLEXITY\n")
    print_complexity_report(complexity, efficiency, model_name)
    
    # 3. Interpretability (for Dense NN)
    if model_type.lower() == 'dense' and vectorizer is not None:
        report.append("\n## 3. INTERPRETABILITY ANALYSIS\n")
        feature_weights = analyze_dense_weights(model, vectorizer, top_n=20)
        if feature_weights:
            print_interpretability_report(feature_weights, model_name, top_n=15)
            
            # Visualize - save to outputs/figures/
            figures_dir = os.path.join(os.path.dirname(save_dir), 'figures')
            save_path = os.path.join(figures_dir, f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
            visualize_feature_importance(feature_weights, model_name, save_path)
    
    # 4. Turing Machine Concepts
    report.append("\n## 4. TURING MACHINE CONCEPTUAL ANALYSIS\n")
    architecture_info = {
        'complexity': complexity,
        'efficiency': efficiency
    }
    turing_analysis = analyze_turing_concepts(model_type, architecture_info)
    print(turing_analysis)
    
    # Save report to file - outputs/metrics/
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
        f.write('\n\n' + turing_analysis)
    
    print(f"\n Full report saved to: {report_path}\n")
    
    # Save metrics as JSON - outputs/metrics/
    metrics_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_metrics.json')
    save_data = {
        'model_name': model_name,
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
        'performance_metrics': metrics,
        'complexity': complexity,
        'efficiency': efficiency
    }
    
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)
    
    print(f" Metrics saved to: {metrics_path}\n")
    
    return '\n'.join(report)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Plot and optionally save a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_training_history(
    history: Any,
    model_name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Keras History object or dict
        model_name: Name of the model
        save_path: Path to save the figure
    """
    if hasattr(history, 'history'):
        hist_dict = history.history
    else:
        hist_dict = history
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    if 'loss' in hist_dict:
        ax1.plot(hist_dict['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in hist_dict:
        ax1.plot(hist_dict['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_name} - Training Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy plot
    if 'accuracy' in hist_dict:
        ax2.plot(hist_dict['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in hist_dict:
        ax2.plot(hist_dict['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title(f'{model_name} - Training Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Training history plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("Available functions:")
    print("  - calculate_performance_metrics()")
    print("  - analyze_model_complexity()")
    print("  - analyze_training_efficiency()")
    print("  - analyze_dense_weights()")
    print("  - analyze_turing_concepts()")
    print("  - generate_model_report()")
    print("  - plot_confusion_matrix()")
    print("  - plot_training_history()")
