# Training and Evaluation Visualizations

After each model training completes, the following comprehensive visualizations and reports are automatically generated:

## 📊 Generated Visualizations

### 1. Training Metrics (`training_metrics.png`)

4-panel plot showing:

- **Training Loss** over epochs
- **Development EER** (Equal Error Rate) with best EER line
- **Development t-DCF** (Tandem Detection Cost Function) with best line
- **Dev vs Eval EER** comparison across epochs

### 2. Final Metrics Summary (`final_metrics.png`)

2-panel plot showing:

- **Final EER** performance with horizontal reference line
- **Final t-DCF** performance with horizontal reference line

### 3. Accuracy Comparison (`accuracy_comparison.png`)

Line plot comparing:

- Development set accuracy over epochs
- Evaluation set accuracy (when available)

### 4. Development Set Confusion Matrix (`confusion_matrix_dev.png`)

Heatmap showing:

- True Positives, True Negatives, False Positives, False Negatives
- Overall accuracy displayed below matrix
- Labels: Fake/Spoof vs Real/Bonafide

### 5. Evaluation Set Confusion Matrix (`confusion_matrix_eval.png`)

Same format as dev set confusion matrix for the final test set

### 6. Development Set ROC Curve (`roc_curve_dev.png`)

ROC curve visualization with:

- True Positive Rate vs False Positive Rate
- AUC (Area Under Curve) score
- EER (Equal Error Rate) point marked
- Random classifier baseline

### 7. Evaluation Set ROC Curve (`roc_curve_eval.png`)

Same format as dev set ROC curve for the final test set

## 📝 Generated Reports

### 1. Metrics CSV (`metrics.csv`)

Epoch-by-epoch metrics in CSV format:

- Epoch number
- Training loss
- Dev EER, t-DCF, accuracy
- Eval EER, t-DCF, accuracy (when available)
- Best dev metrics tracking

### 2. Metrics JSON (`metrics.json`)

Complete metrics in JSON format for programmatic access

### 3. Classification Report - Dev (`classification_report_dev.txt`)

Detailed sklearn classification report including:

- Precision, Recall, F1-score for each class
- Support (number of samples)
- Macro and weighted averages

### 4. Classification Report - Eval (`classification_report_eval.txt`)

Same format as dev report for the evaluation set

### 5. Metrics Summary (`metrics_summary.txt`)

Human-readable text summary with:

- Configuration details (epochs, batch size, features, augmentation)
- Final results (EER, t-DCF, accuracy)
- Best development metrics with epoch numbers
- Training statistics (loss progression, averages)

### 6. Final Summary (`final_summary.txt`)

Comprehensive training summary with:

- Training metrics (initial/final/min loss)
- Development set best metrics
- Evaluation set final metrics (EER, ROC AUC, accuracy)

## 📂 Output Location

All visualizations and reports are saved to:

```
exp_result/<model_tag>/metrics/
```

Where `<model_tag>` is based on:

- Model architecture
- Dataset name
- Feature type
- Augmentation settings
- Timestamp

## 🎯 Usage

After training completes, you'll see:

```
================================================================================
                    GENERATING COMPREHENSIVE VISUALIZATIONS
================================================================================

📊 Collecting predictions for visualization...

🔍 Development Set:
✓ Confusion matrix saved to ...
✓ ROC curve saved to ...
✓ Classification report saved to ...

🔍 Evaluation Set:
✓ Confusion matrix saved to ...
✓ ROC curve saved to ...
✓ Classification report saved to ...

✓ Accuracy comparison saved to ...

================================================================================
                        FINAL TRAINING SUMMARY
================================================================================
[Detailed metrics displayed here]

✅ All visualizations generated successfully!
================================================================================
```

## 📈 Interpreting Results

### Confusion Matrix

- **Diagonal elements** (top-left, bottom-right): Correct predictions
- **Off-diagonal elements**: Misclassifications
- High accuracy = large diagonal values

### ROC Curve

- **Closer to top-left corner** = Better performance
- **AUC close to 1.0** = Excellent discrimination
- **EER point**: Where false positive rate = false negative rate

### Training Curves

- **Decreasing loss**: Model is learning
- **Dev EER plateau**: Model has converged
- **Gap between dev/eval**: Check for overfitting

## 🔧 Customization

To modify visualizations, edit:

- `metrics.py` - Core visualization functions
- `main.py` - When/how visualizations are called
