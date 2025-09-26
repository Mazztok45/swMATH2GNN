# Appendix: Hyperparameter Settings

## Model Architecture Parameters
- Input dimension: 768 (for title features) / 2048 (for MSC features)
- Hidden layer size: 512 neurons
- Activation function: ReLU
- Output dimension: Number of classes in multi-label classification
- Loss function: BCEWithLogitsLoss with class weights

## Training Hyperparameters
- Learning rate: 0.001
- Batch size: 128
- Epochs: 50
- L2 regularization (weight decay): 1e-4
- Optimizer: Adam
- Train-validation split: 80%-20%
- Random seed: 42

## Class Weight Calculation
- Positive class weights: Computed as `negative count / positive count` per label
- Default weight: 1.0 for labels with no positive samples

## Threshold Optimization
- Threshold search range: 0.0 to 1.0 with step size 0.01
- Optimization metric: F1-score (binary)
- Default threshold: 0.5 when optimization not possible
