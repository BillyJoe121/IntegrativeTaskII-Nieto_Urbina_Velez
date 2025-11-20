# Output Directory Structure

This document describes how the evaluation module organizes its output files.

## Directory Structure

```
outputs/
├── figures/          # Plots, word clouds, charts used in report
│   ├── dense_neural_network_feature_importance.png
│   ├── dense_neural_network_confusion_matrix.png
│   ├── dense_neural_network_training_history.png
│   ├── rnn_confusion_matrix.png
│   ├── lstm_confusion_matrix.png
│   └── ... (other visualizations)
│
├── metrics/          # Model performance metrics or confusion matrices (data)
│   ├── dense_neural_network_evaluation_report.txt
│   ├── dense_neural_network_metrics.json
│   ├── rnn_evaluation_report.txt
│   ├── rnn_metrics.json
│   ├── lstm_evaluation_report.txt
│   ├── lstm_metrics.json
│   └── ... (other metrics files)
│
└── saved_models/     # Trained model weights (if small enough)
    └── ... (model files if saved)
```

## File Types by Directory

### `outputs/figures/` - Visual outputs
- **Feature importance plots** (`*_feature_importance.png`)
- **Confusion matrices** (`*_confusion_matrix.png`)
- **Training history curves** (`*_training_history.png`)
- **Word clouds** (if generated)
- **Comparison charts** (for final analysis)

### `outputs/metrics/` - Data and reports
- **Evaluation reports** (`*_evaluation_report.txt`) - Full text analysis
- **Metrics JSON** (`*_metrics.json`) - Machine-readable performance data
- **Confusion matrix data** (embedded in JSON)

### `outputs/saved_models/` - Model artifacts
- **Trained model weights** (`.h5`, `.keras` files)
- **Vectorizers** (`.pkl` files)
- Only if models are small enough to version control

## Usage in evaluate.py

The module automatically organizes files:

```python
# When calling generate_model_report():
generate_model_report(
    model=dnn_model,
    metrics=metrics,
    history=history,
    model_name="Dense Neural Network",
    model_type='dense',
    vectorizer=vectorizer,
    save_dir='../outputs/metrics'  # Default location
)

# This creates:
# - ../outputs/metrics/dense_neural_network_evaluation_report.txt
# - ../outputs/metrics/dense_neural_network_metrics.json
# - ../outputs/figures/dense_neural_network_feature_importance.png

# For confusion matrices:
plot_confusion_matrix(
    y_test, y_pred,
    "Dense Neural Network",
    save_path='../outputs/figures/dense_neural_network_confusion_matrix.png'
)

# For training history:
plot_training_history(
    history,
    "Dense Neural Network",
    save_path='../outputs/figures/dense_neural_network_training_history.png'
)
```

## Notes

- All directories are created automatically if they don't exist
- File names use lowercase with underscores (e.g., `dense_neural_network`)
- Visualizations always go to `outputs/figures/`
- Metrics and reports always go to `outputs/metrics/`
