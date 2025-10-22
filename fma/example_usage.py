"""
Example script demonstrating how to use the FMA data loader.

This script shows various ways to load and use the FMA dataset
for machine learning preparation.
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add the current directory to the path so we can import the data loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import FMADataLoader


class MusicGenreClassifier(nn.Module):
    """
    PyTorch neural network for music genre classification.
    """
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2):
        super(MusicGenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_pytorch_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device='cpu'):
    """
    Train the PyTorch model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    
    model.to(device)
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }


def evaluate_pytorch_model(model, data_loader, device='cpu'):
    """
    Evaluate the PyTorch model and return predictions and probabilities.
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
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


def load_saved_model(model_path):
    """
    Load a saved PyTorch model from .pth file.
    
    Args:
        model_path (str): Path to the .pth file
        
    Returns:
        dict: Dictionary containing the loaded model and metadata
    """
    # Load the saved data
    saved_data = torch.load(model_path, map_location='cpu')
    
    # Recreate the model
    model = MusicGenreClassifier(
        input_size=saved_data['input_size'],
        hidden_size=saved_data['hidden_size'],
        num_classes=saved_data['num_classes']
    )
    
    # Load the trained weights
    model.load_state_dict(saved_data['model_state_dict'])
    
    print(f"Model loaded from: {model_path}")
    print(f"Test accuracy when saved: {saved_data['test_accuracy']:.4f}")
    print(f"Available genres: {saved_data['genre_names']}")
    
    return {
        'model': model,
        'genre_names': saved_data['genre_names'],
        'feature_names': saved_data['feature_names'],
        'test_accuracy': saved_data['test_accuracy'],
        'training_history': saved_data.get('training_history', None)
    }


def get_metadata_path():
    """Get the correct path to the metadata directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), 'fma_metadata')


def demonstrate_basic_loading():
    """
    Demonstrate basic data loading functionality.
    """
    print("=" * 60)
    print("BASIC DATA LOADING DEMONSTRATION")
    print("=" * 60)
    
    # Get the correct metadata path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(os.path.dirname(script_dir), 'fma_metadata')
    
    # Initialize the loader
    loader = FMADataLoader(metadata_path=metadata_path)
    
    # Load basic dataset information
    print("\n1. Loading metadata...")
    metadata = loader.load_metadata()
    
    # Show some basic statistics
    print(f"\nDataset Statistics:")
    print(f"Total tracks: {metadata['tracks'].shape[0]:,}")
    print(f"Total features per track: {metadata['features'].shape[1]:,}")
    print(f"Total genres: {metadata['genres'].shape[0]:,}")
    
    # Show available subsets
    subset_counts = metadata['tracks']['set', 'subset'].value_counts()
    print(f"\nAvailable subsets:")
    for subset, count in subset_counts.items():
        print(f"  {subset}: {count:,} tracks")
    
    # Show top genres
    top_genres = metadata['tracks']['track', 'genre_top'].value_counts().head(8)
    print(f"\nTop 8 genres:")
    for genre, count in top_genres.items():
        print(f"  {genre}: {count:,} tracks")


def demonstrate_feature_extraction():
    """
    Demonstrate different ways to extract features and labels.
    """
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION DEMONSTRATION")
    print("=" * 60)
    
    loader = FMADataLoader(metadata_path=get_metadata_path())
    
    # Example 1: Small dataset with all features
    print("\n1. Loading small dataset (all features)...")
    data = loader.get_first_n_tracks(n=500, subset='small')
    
    print(f"Feature matrix shape: {data['X'].shape}")
    print(f"Labels shape: {data['y'].shape}")
    print(f"Number of unique genres: {len(data['genre_names'])}")
    print(f"Genres: {data['genre_names']}")
    
    # Example 2: Specific features only
    print("\n2. Loading with specific features (MFCC only)...")
    data_mfcc = loader.get_first_n_tracks(n=500, subset='small', 
                                         feature_columns='mfcc')
    
    print(f"MFCC feature matrix shape: {data_mfcc['X'].shape}")
    print(f"Number of MFCC features: {data_mfcc['X'].shape[1]}")
    
    # Example 3: Multiple specific features
    print("\n3. Loading with multiple feature types...")
    feature_cols = ['mfcc', 'spectral_contrast', 'chroma_cens']
    data_multi = loader.get_first_n_tracks(n=500, subset='small',
                                          feature_columns=feature_cols)
    
    print(f"Multi-feature matrix shape: {data_multi['X'].shape}")
    print(f"Selected features: {feature_cols}")
    
    # Example 4: Multi-label classification
    print("\n4. Loading for multi-label classification...")
    data_multilabel = loader.get_first_n_tracks(n=200, subset='small',
                                               multi_label=True)
    
    print(f"Multi-label matrix shape: {data_multilabel['X'].shape}")
    print(f"Multi-label y shape: {data_multilabel['y'].shape}")
    print(f"Number of possible genres: {len(data_multilabel['genre_names'])}")


def demonstrate_train_test_split():
    """
    Demonstrate the train/test split functionality.
    """
    print("\n" + "=" * 60)
    print("TRAIN/TEST SPLIT DEMONSTRATION")
    print("=" * 60)
    
    loader = FMADataLoader(metadata_path=get_metadata_path())
    
    # Get train/test split
    print("\n1. Getting train/validation/test split...")
    split_data = loader.get_train_test_split(
        n=1000, 
        subset='small',
        feature_columns=['mfcc', 'spectral_contrast'],
        standardize=True,
        shuffle_data=True
    )
    
    print(f"\nSplit sizes:")
    print(f"  Training: {split_data['X_train'].shape[0]} samples")
    print(f"  Validation: {split_data['X_val'].shape[0]} samples") 
    print(f"  Test: {split_data['X_test'].shape[0]} samples")
    
    print(f"\nFeature information:")
    print(f"  Number of features: {split_data['X_train'].shape[1]}")
    print(f"  Features are standardized: {split_data['scaler'] is not None}")
    
    print(f"\nLabel information:")
    print(f"  Number of classes: {len(split_data['genre_names'])}")
    print(f"  Classes: {split_data['genre_names']}")
    
    # Show some statistics about the features
    print(f"\nFeature statistics (after standardization):")
    print(f"  Training data mean: {split_data['X_train'].mean():.6f}")
    print(f"  Training data std: {split_data['X_train'].std():.6f}")
    
    # Show class distribution in training set
    unique, counts = np.unique(split_data['y_train'], return_counts=True)
    print(f"\nClass distribution in training set:")
    for class_idx, count in zip(unique, counts):
        class_name = split_data['genre_names'][class_idx]
        print(f"  {class_name}: {count} samples")


def demonstrate_metadata_access():
    """
    Demonstrate accessing track metadata.
    """
    print("\n" + "=" * 60)
    print("METADATA ACCESS DEMONSTRATION")
    print("=" * 60)
    
    loader = FMADataLoader(metadata_path=get_metadata_path())
    
    # Get data with metadata
    print("\n1. Loading tracks with metadata...")
    data = loader.get_first_n_tracks(n=50, subset='small')
    
    if data['metadata'] is not None:
        print(f"\nAvailable metadata columns: {list(data['metadata'].columns)}")
        print(f"\nFirst 5 tracks metadata:")
        print(data['metadata'].head())
        
        # Show some interesting statistics
        if ('track', 'duration') in data['metadata'].columns:
            durations = data['metadata'][('track', 'duration')]
            print(f"\nTrack duration statistics:")
            print(f"  Mean duration: {durations.mean():.1f} seconds")
            print(f"  Min duration: {durations.min():.1f} seconds")
            print(f"  Max duration: {durations.max():.1f} seconds")
        
        if ('track', 'listens') in data['metadata'].columns:
            listens = data['metadata'][('track', 'listens')]
            print(f"\nTrack popularity statistics:")
            print(f"  Mean listens: {listens.mean():.1f}")
            print(f"  Min listens: {listens.min()}")
            print(f"  Max listens: {listens.max()}")
    else:
        print("No metadata available for selected tracks.")


def demonstrate_neural_network_training():
    """
    Demonstrate training a PyTorch neural network for genre prediction.
    """
    print("\n" + "=" * 60)
    print("PYTORCH NEURAL NETWORK TRAINING DEMONSTRATION")
    print("=" * 60)
    
    loader = FMADataLoader(metadata_path=get_metadata_path())
    
    print("\n1. Loading and preparing data for neural network training...")
    
    # Get train/test split with standardized features
    split_data = loader.get_train_test_split(
        n=2000,  # Use 2000 tracks for faster training
        subset='small',
        feature_columns=['mfcc', 'spectral_contrast', 'chroma_cens'],  # Select key features
        standardize=True,
        shuffle_data=True
    )
    
    X_train = split_data['X_train']
    X_val = split_data['X_val']
    X_test = split_data['X_test']
    y_train = split_data['y_train']
    y_val = split_data['y_val']
    y_test = split_data['y_test']
    
    print(f"\nDataset prepared:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features per sample: {X_train.shape[1]}")
    print(f"  Number of genres: {len(split_data['genre_names'])}")
    print(f"  Genres: {split_data['genre_names']}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create PyTorch model
    print("\n2. Creating PyTorch neural network...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_train.shape[1]
    hidden_size = 100
    num_classes = len(split_data['genre_names'])
    
    model = MusicGenreClassifier(input_size, hidden_size, num_classes)
    
    print(f"Model created:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Output size (genres): {num_classes}")
    print(f"  Device: {device}")
    
    # Train the model
    print("\n3. Training PyTorch neural network...")
    start_time = time.time()
    
    training_history = train_pytorch_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=50,
        learning_rate=0.001,
        device=device
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate model performance
    print("\n4. Evaluating model performance...")
    
    # Training accuracy
    train_pred, train_probs, train_labels = evaluate_pytorch_model(model, train_loader, device)
    train_accuracy = accuracy_score(train_labels, train_pred)
    
    # Validation accuracy
    val_pred, val_probs, val_labels = evaluate_pytorch_model(model, val_loader, device)
    val_accuracy = accuracy_score(val_labels, val_pred)
    
    # Test accuracy
    test_pred, test_probs, test_labels = evaluate_pytorch_model(model, test_loader, device)
    test_accuracy = accuracy_score(test_labels, test_pred)
    
    print(f"\nModel Performance:")
    print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Save the model
    print("\n5. Saving PyTorch model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'music_genre_classifier.pth')
    
    # Save model state dict along with metadata
    model_data = {
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_classes': num_classes,
        'genre_names': split_data['genre_names'],
        'feature_names': split_data['feature_names'],
        'test_accuracy': test_accuracy,
        'training_history': training_history
    }
    
    torch.save(model_data, model_path)
    print(f"Model saved to: {model_path}")
    
    # Detailed classification report
    print(f"\n6. Detailed Classification Report (Test Set):")
    print("-" * 50)
    class_report = classification_report(
        test_labels, 
        test_pred, 
        target_names=split_data['genre_names'],
        digits=3
    )
    print(class_report)
    
    # Confusion matrix
    print(f"\n7. Confusion Matrix:")
    print("-" * 30)
    cm = confusion_matrix(test_labels, test_pred)
    
    # Print confusion matrix in a readable format
    genre_names = split_data['genre_names']
    print(f"{'Actual // Predicted':<20}", end="")
    for genre in genre_names:
        print(f"{genre:<8}", end="")
    print()
    
    for i, actual_genre in enumerate(genre_names):
        print(f"{actual_genre:<20}", end="")
        for j in range(len(genre_names)):
            print(f"{cm[i,j]:<8}", end="")
        print()
    
    # Show some prediction examples
    print(f"\n8. Sample Predictions:")
    print("-" * 40)
    
    # Get some test samples
    num_samples = min(10, len(test_labels))
    sample_indices = np.random.choice(len(test_labels), num_samples, replace=False)
    
    for idx in sample_indices:
        actual_genre = split_data['genre_names'][test_labels[idx]]
        predicted_genre = split_data['genre_names'][test_pred[idx]]
        
        # Get prediction confidence
        confidence = test_probs[idx].max()
        
        status = "✓" if actual_genre == predicted_genre else "✗"
        print(f"  {status} Actual: {actual_genre:<12} | Predicted: {predicted_genre:<12} | Confidence: {confidence:.3f}")
    
    # Feature importance analysis (using weights from first layer)
    print(f"\n9. Neural Network Analysis:")
    print("-" * 30)
    
    # Get weights from input to first hidden layer
    with torch.no_grad():
        input_weights = model.fc1.weight.data.cpu().numpy()  # Shape: (hidden_size, input_size)
        
        # Calculate average absolute weight per feature
        feature_importance = np.mean(np.abs(input_weights), axis=0)  # Average over hidden units
        
        # Get top 10 most important features
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        
        print("Top 10 most important features (by average absolute weight):")
        for i, feat_idx in enumerate(top_features_idx):
            importance = feature_importance[feat_idx]
            feature_name = split_data['feature_names'][feat_idx] if feat_idx < len(split_data['feature_names']) else f"Feature_{feat_idx}"
            print(f"  {i+1:2d}. {feature_name}: {importance:.4f}")
    
    print(f"\nNetwork architecture:")
    print(f"  Input layer: {input_size} features")
    print(f"  Hidden layer: {hidden_size} neurons (ReLU activation)")
    print(f"  Output layer: {num_classes} genres")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return {
        'model': model,
        'model_path': model_path,
        'test_accuracy': test_accuracy,
        'feature_names': split_data['feature_names'],
        'genre_names': split_data['genre_names'],
        'confusion_matrix': cm,
        'training_history': training_history
    }


def main():
    """
    Run all demonstration examples.
    """
    print("FMA Dataset Loader - Comprehensive Examples")
    print("This script demonstrates various ways to use the data loader")
    print("for preparing FMA dataset for machine learning.\n")
    
    try:
        # Run all demonstrations
        demonstrate_basic_loading()
        demonstrate_feature_extraction()
        demonstrate_train_test_split()
        demonstrate_metadata_access()
        demonstrate_neural_network_training()
        
        print("\n" + "=" * 60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe data loader is ready to use for your machine learning projects.")
        print("You can now import FMADataLoader and use it to load data for:")
        print("  - Single-label genre classification")
        print("  - Multi-label genre classification") 
        print("  - Feature selection and extraction")
        print("  - Train/validation/test splitting")
        print("  - Data standardization")
        print("  - PyTorch neural network training for music genre prediction")
        print("  - Model saving in .pth format for later use")
        
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find required files.")
        print(f"Please make sure the FMA metadata files are in the 'fma_metadata' directory.")
        print(f"Expected path: {get_metadata_path()}")
        print(f"Required files: tracks.csv, features.csv, genres.csv, echonest.csv")
        print(f"Error details: {e}")
        
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")
        print("Please check your installation and file paths.")


if __name__ == "__main__":
    main()