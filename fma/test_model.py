#!/usr/bin/env python3
"""
Music Genre Classifier - Model Testing Script

Loads a trained PyTorch model and evaluates it on test data.
Automatically infers model architecture from the .pth file and provides
comprehensive evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import FMADataLoader


# ============================================================================
# USER CONFIGURATION - Modify these settings as needed
# ============================================================================

CONFIG = {
    'model_path': 'best_model.pth',           # Path to your trained model
    'metadata_path': '../fma_metadata',        # Path to FMA metadata directory
    'test_subset': 'large',                   # Dataset subset: 'small', 'medium', or 'large'
    'num_test_samples': 100000,                  # Number of samples to test
    'batch_size': 32,                          # Batch size for evaluation
    'num_sample_predictions': 20,              # Number of sample predictions to display
}

# ============================================================================

class MusicGenreClassifier(nn.Module):
    """Neural network that automatically infers architecture from state dict."""
    
    def __init__(self, state_dict=None):
        super(MusicGenreClassifier, self).__init__()
        
        if state_dict is None:
            raise ValueError("State dictionary must be provided to infer model dimensions.")
            
        self.input_size, self.hidden_size, self.num_classes = self._infer_dimensions(state_dict)
        
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )
    
    def _infer_dimensions(self, state_dict):
        first_layer = state_dict['net.0.weight']
        last_layer = state_dict['net.4.weight']
        return first_layer.shape[1], first_layer.shape[0], last_layer.shape[0]
        
    def forward(self, x):
        return self.net(x)


def load_model(model_path):
    """Load trained model and infer architecture from .pth file."""
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    
    loaded_data = torch.load(model_path, map_location='cpu')
    state_dict = loaded_data['model_state_dict']
    
    model = MusicGenreClassifier(state_dict=state_dict)
    model.load_state_dict(state_dict)
    
    print(f"✓ Model loaded successfully!")
    print(f"  Input: {model.input_size} features")
    print(f"  Hidden: {model.hidden_size} neurons x2")
    print(f"  Output: {model.num_classes} classes")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, loaded_data


def evaluate_model(model, data_loader, device='cpu'):
    """Evaluate model and return predictions, probabilities, and labels."""
    model.eval()
    all_predictions, all_probabilities, all_labels = [], [], []
    
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)


def print_classification_metrics(y_true, y_pred, y_probs, genre_names):
    """
    Print comprehensive classification metrics.
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION METRICS")
    print("=" * 60)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Get unique labels actually present in the data
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    present_genre_names = [genre_names[i] for i in unique_labels if i < len(genre_names)]
    
    # Detailed classification report
    print("\n" + "-" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("-" * 60)
    class_report = classification_report(
        y_true, 
        y_pred, 
        labels=unique_labels,
        target_names=present_genre_names,
        digits=4,
        zero_division=0
    )
    print(class_report)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=unique_labels, zero_division=0
    )
    
    print("\n" + "-" * 60)
    print("PER-CLASS PERFORMANCE SUMMARY")
    print("-" * 60)
    print(f"{'Genre':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for i, label_idx in enumerate(unique_labels):
        genre = genre_names[label_idx] if label_idx < len(genre_names) else f"Unknown_{label_idx}"
        print(f"{genre:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    print("-" * 60)
    print(f"{'Macro Avg':<15} {precision_macro:<12.4f} {recall_macro:<12.4f} {f1_macro:<12.4f}")
    print(f"{'Weighted Avg':<15} {precision_weighted:<12.4f} {recall_weighted:<12.4f} {f1_weighted:<12.4f}")
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }


def print_confusion_matrix(y_true, y_pred, genre_names):
    """
    Print confusion matrix in a readable format.
    """
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    
    # Get unique labels actually present
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    present_genre_names = [genre_names[i] for i in unique_labels if i < len(genre_names)]
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Print header
    print(f"\n{'Actual // Predicted':<20}", end="")
    for genre in present_genre_names:
        print(f"{genre[:6]:<8}", end="")
    print()
    print("-" * (20 + len(present_genre_names) * 8))
    
    # Print each row
    for i, actual_genre in enumerate(present_genre_names):
        print(f"{actual_genre:<20}", end="")
        for j in range(len(present_genre_names)):
            print(f"{cm[i,j]:<8}", end="")
        print()
    
    return cm


def print_sample_predictions(y_true, y_pred, y_probs, genre_names, num_samples=20):
    """
    Print sample predictions with confidence scores.
    """
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Get random samples
    num_samples = min(num_samples, len(y_true))
    sample_indices = np.random.choice(len(y_true), num_samples, replace=False)
    
    # Count correct and incorrect
    correct_count = 0
    incorrect_count = 0
    
    print(f"\nShowing {num_samples} random predictions:")
    print("-" * 60)
    
    for idx in sample_indices:
        actual_genre = genre_names[y_true[idx]]
        predicted_genre = genre_names[y_pred[idx]]
        confidence = y_probs[idx].max()
        
        is_correct = actual_genre == predicted_genre
        status = "CORRECT" if is_correct else "WRONG  "
        symbol = "✓" if is_correct else "✗"
        
        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1
        
        print(f"  {symbol} [{status}] Actual: {actual_genre:<12} | Predicted: {predicted_genre:<12} | Confidence: {confidence:.4f}")
    
    print("-" * 60)
    print(f"In sample: {correct_count} correct, {incorrect_count} incorrect")
    print(f"Sample accuracy: {correct_count/num_samples:.4f} ({correct_count/num_samples*100:.2f}%)")


def analyze_prediction_confidence(y_true, y_pred, y_probs):
    """
    Analyze prediction confidence statistics.
    """
    print("\n" + "=" * 60)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("=" * 60)
    
    # Get max probability for each prediction
    max_probs = y_probs.max(axis=1)
    
    # Split into correct and incorrect predictions
    correct_mask = (y_true == y_pred)
    correct_probs = max_probs[correct_mask]
    incorrect_probs = max_probs[~correct_mask]
    
    print(f"\nAll Predictions:")
    print(f"  Mean confidence: {max_probs.mean():.4f}")
    print(f"  Median confidence: {np.median(max_probs):.4f}")
    print(f"  Min confidence: {max_probs.min():.4f}")
    print(f"  Max confidence: {max_probs.max():.4f}")
    print(f"  Std deviation: {max_probs.std():.4f}")
    
    if len(correct_probs) > 0:
        print(f"\nCorrect Predictions:")
        print(f"  Count: {len(correct_probs)}")
        print(f"  Mean confidence: {correct_probs.mean():.4f}")
        print(f"  Median confidence: {np.median(correct_probs):.4f}")
        print(f"  Min confidence: {correct_probs.min():.4f}")
        print(f"  Max confidence: {correct_probs.max():.4f}")
    
    if len(incorrect_probs) > 0:
        print(f"\nIncorrect Predictions:")
        print(f"  Count: {len(incorrect_probs)}")
        print(f"  Mean confidence: {incorrect_probs.mean():.4f}")
        print(f"  Median confidence: {np.median(incorrect_probs):.4f}")
        print(f"  Min confidence: {incorrect_probs.min():.4f}")
        print(f"  Max confidence: {incorrect_probs.max():.4f}")
    
    # Confidence thresholds
    print(f"\nPredictions by Confidence Level:")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        high_conf_mask = max_probs >= threshold
        high_conf_count = high_conf_mask.sum()
        high_conf_accuracy = (y_true[high_conf_mask] == y_pred[high_conf_mask]).mean() if high_conf_count > 0 else 0
        print(f"  Confidence >= {threshold:.1f}: {high_conf_count:4d} predictions ({high_conf_count/len(y_true)*100:5.2f}%), Accuracy: {high_conf_accuracy:.4f}")


def print_model_summary(model, loaded_data):
    """
    Print model architecture and metadata summary.
    """
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    
    print(f"\nModel Architecture:")
    print(model)
    
    print(f"\nModel Dimensions:")
    print(f"  Input layer: {model.input_size} features")
    print(f"  Hidden layer 1: {model.hidden_size} neurons (ReLU)")
    print(f"  Hidden layer 2: {model.hidden_size} neurons (ReLU)")
    print(f"  Output layer: {model.num_classes} classes")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter Count:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Check for metadata in loaded data
    if 'genre_names' in loaded_data:
        print(f"\nGenre Labels ({len(loaded_data['genre_names'])}):")
        for i, genre in enumerate(loaded_data['genre_names']):
            print(f"  {i}: {genre}")
    
    if 'test_accuracy' in loaded_data:
        print(f"\nStored Test Accuracy: {loaded_data['test_accuracy']:.4f} ({loaded_data['test_accuracy']*100:.2f}%)")
    
    if 'epoch' in loaded_data:
        print(f"Training Epoch: {loaded_data['epoch']}")
    
    

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MUSIC GENRE CLASSIFIER - MODEL TESTING")
    print("=" * 60)
    
    # Load the model
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
    loaded_model, loaded_data = load_model(model_path)
    
    # Print model summary
    print_model_summary(loaded_model, loaded_data)
    
    # Load test data
    print("\n" + "=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)
    
    loader = FMADataLoader("../fma_metadata")

    print(f"\nLoading test tracks from {CONFIG['test_subset']} subset...")
    test_data = loader.get_first_n_tracks(
        n=CONFIG['num_test_samples'], 
        subset=CONFIG['test_subset'],
        feature_columns=None, 
        multi_label=False, 
        include_echonest=False
    )
    
    X_test = test_data['X']
    y_test = test_data['y']
    test_genre_names = test_data['genre_names']
    
    print(f"\nTest Data Loaded:")
    print(f"  Number of samples: {X_test.shape[0]}")
    print(f"  Number of features: {X_test.shape[1]}")
    print(f"  Number of genres: {len(test_genre_names)}")
    print(f"  Genres: {test_genre_names}")
    
    # Get the model's training genre labels if available
    if 'genre_names' in loaded_data:
        model_genre_names = loaded_data['genre_names']
        print(f"\nModel was trained on {len(model_genre_names)} genres:")
        print(f"  {model_genre_names}")
    else:
        # Model file doesn't have genre names, but we know it was trained on 16 genres from 'medium' subset
        print(f"\nNo genre names in model file. Assuming model was trained on 16 genres from 'medium' subset.")
        model_genre_names = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic', 
                            'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 
                            'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']
        print(f"  {model_genre_names}")
        
        # Verify this matches the model's output size
        if len(model_genre_names) != loaded_model.num_classes:
            print(f"\nWARNING: Genre count mismatch!")
            print(f"  Assumed genres: {len(model_genre_names)}")
            print(f"  Model output size: {loaded_model.num_classes}")
    
    # Check if test genres match model genres
    if test_genre_names != model_genre_names:
        print(f"\n{'!'*60}")
        print("WARNING: Genre mismatch detected!")
        print(f"{'!'*60}")
        print(f"Test data has {len(test_genre_names)} genres but model expects {len(model_genre_names)} genres")
        print("\nThis mismatch will cause incorrect predictions!")
        print("The model needs data with the same genre labels it was trained on.")
        
        # Try to create a mapping
        print("\nAttempting to filter and remap genres...")
        
        # Find which test genres are in the model's training genres
        valid_indices = []
        genre_mapping = {}  # Maps test label index to model label index
        
        for test_idx, test_genre in enumerate(test_genre_names):
            if test_genre in model_genre_names:
                model_idx = model_genre_names.index(test_genre)
                genre_mapping[test_idx] = model_idx
                print(f"  Test genre '{test_genre}' (idx {test_idx}) -> Model genre (idx {model_idx})")
        
        # Filter test data to only include samples with matching genres
        print(f"\nFiltering test data to only include matching genres...")
        original_size = len(y_test)
        
        # Create mask for valid samples
        valid_mask = np.isin(y_test, list(genre_mapping.keys()))
        X_test = X_test[valid_mask]
        y_test_old = y_test[valid_mask]
        
        # Remap labels to model's genre indices
        y_test = np.array([genre_mapping[label] for label in y_test_old])
        
        print(f"  Filtered from {original_size} to {len(y_test)} samples")
        print(f"  Using {len(genre_mapping)} matching genres")
        
        # Debug: show some remapped labels
        print(f"\nDebug - Sample label remapping:")
        for i in range(min(5, len(y_test_old))):
            old_label = y_test_old[i]
            new_label = y_test[i]
            old_genre = test_genre_names[old_label]
            new_genre = model_genre_names[new_label]
            print(f"  Sample {i}: {old_label} ('{old_genre}') -> {new_label} ('{new_genre}')")
        
        # Use model's genre names for evaluation
        genre_names = model_genre_names
    else:
        print("\nGenre labels match! Test data and model use the same genres.")
        genre_names = test_genre_names
    
    # Verify feature dimensions match
    if X_test.shape[1] != loaded_model.input_size:
        print(f"\nWARNING: Feature dimension mismatch!")
        print(f"  Model expects: {loaded_model.input_size} features")
        print(f"  Data has: {X_test.shape[1]} features")
    
    print(f"\nFinal Test Set:")
    print(f"  Number of samples: {X_test.shape[0]}")
    print(f"  Number of features: {X_test.shape[1]}")
    print(f"  Number of genres: {len(genre_names)}")
    
    # Assuming the model was trained on standardized features (mean=0, std=1)
    print("Applying standardization to test data...")
    
    scaler = StandardScaler()
    X_test_standardized = scaler.fit_transform(X_test)
    
    print(f"\nBefore standardization:")
    print(f"  Mean: {X_test.mean():.6f}, Std: {X_test.std():.6f}")
    print(f"After standardization:")
    print(f"  Mean: {X_test_standardized.mean():.6f}, Std: {X_test_standardized.std():.6f}")
    
    # Use standardized data
    X_test = X_test_standardized
    
    # Convert to PyTorch tensors
    print("\nPreparing data for evaluation...")
    x_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loader
    batch_size = 32
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate the model
    print("\n" + "=" * 60)
    print("EVALUATING MODEL ON TEST DATA")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    loaded_model.to(device)
    loaded_model.eval()
    
    print("Running predictions...")
    y_pred, y_probs, y_true = evaluate_model(loaded_model, test_loader, device)
    
    # Print all metrics and results
    metrics = print_classification_metrics(y_true, y_pred, y_probs, genre_names)
    cm = print_confusion_matrix(y_true, y_pred, genre_names)
    print_sample_predictions(y_true, y_pred, y_probs, genre_names, num_samples=20)
    analyze_prediction_confidence(y_true, y_pred, y_probs)
    
    # Final summary
    print("\n" + "=" * 60)
    print("TEST COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"\nTest Set Performance:")
    print(f"  Total samples tested: {len(y_true)}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1-Score: {metrics['f1_weighted']:.4f}")
    

