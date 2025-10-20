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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Add the current directory to the path so we can import the data loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import FMADataLoader


def get_metadata_path():
    """Get the correct path to the metadata directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")
    print(f"Metadata directory: {os.path.join(script_dir, 'fma_metadata')}")
    return os.path.join(script_dir, 'fma_metadata')  # Changed: join with script_dir instead of dirname(script_dir)


def demonstrate_basic_loading():
    """
    Demonstrate basic data loading functionality.
    """
    print("=" * 60)
    print("BASIC DATA LOADING DEMONSTRATION")
    print("=" * 60)
    
    # Get the correct metadata path
    metadata_path = get_metadata_path()
    
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
    Demonstrate training a simple one-layer neural network for genre prediction.
    """
    print("\n" + "=" * 60)
    print("NEURAL NETWORK TRAINING DEMONSTRATION")
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
    
    # Create and train a simple one-layer neural network
    print("\n2. Creating and training one-layer neural network...")
    
    # MLPClassifier with one hidden layer
    nn_model = MLPClassifier(
        hidden_layer_sizes=(100,),  # One layer with 100 neurons
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        learning_rate='adaptive',
        max_iter=500,
        random_state=42,
        verbose=True  # Show training progress
    )
    
    # Train the model
    print("Training neural network...")
    start_time = time.time()
    nn_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Number of iterations: {nn_model.n_iter_}")
    print(f"Final loss: {nn_model.loss_:.4f}")
    
    # Make predictions
    print("\n3. Evaluating model performance...")
    
    # Training accuracy
    train_pred = nn_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    # Validation accuracy
    val_pred = nn_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    # Test accuracy
    test_pred = nn_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"\nModel Performance:")
    print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Detailed classification report
    print(f"\n4. Detailed Classification Report (Test Set):")
    print("-" * 50)
    class_report = classification_report(
        y_test, 
        test_pred, 
        target_names=split_data['genre_names'],
        digits=3
    )
    print(class_report)
    
    # Confusion matrix
    print(f"\n5. Confusion Matrix:")
    print("-" * 30)
    cm = confusion_matrix(y_test, test_pred)
    
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
    print(f"\n6. Sample Predictions:")
    print("-" * 40)
    
    # Get some test samples
    num_samples = min(10, len(y_test))
    sample_indices = np.random.choice(len(y_test), num_samples, replace=False)
    
    for idx in sample_indices:
        actual_genre = split_data['genre_names'][y_test[idx]]
        predicted_genre = split_data['genre_names'][test_pred[idx]]
        
        # Get prediction probabilities
        probabilities = nn_model.predict_proba(X_test[idx:idx+1])[0]
        confidence = probabilities.max()
        
        status = "✓" if actual_genre == predicted_genre else "✗"
        print(f"  {status} Actual: {actual_genre:<12} | Predicted: {predicted_genre:<12} | Confidence: {confidence:.3f}")
    
    # Feature importance analysis (using weights from first layer)
    print(f"\n7. Neural Network Analysis:")
    print("-" * 30)
    
    # Get weights from input to first hidden layer
    input_weights = nn_model.coefs_[0]  # Shape: (n_features, n_neurons)
    
    # Calculate average absolute weight per feature
    feature_importance = np.mean(np.abs(input_weights), axis=1)
    
    # Get top 10 most important features
    top_features_idx = np.argsort(feature_importance)[-10:][::-1]
    
    print("Top 10 most important features (by average absolute weight):")
    for i, feat_idx in enumerate(top_features_idx):
        importance = feature_importance[feat_idx]
        feature_name = split_data['feature_names'][feat_idx] if feat_idx < len(split_data['feature_names']) else f"Feature_{feat_idx}"
        print(f"  {i+1:2d}. {feature_name}: {importance:.4f}")
    
    print(f"\nNetwork architecture:")
    print(f"  Input layer: {X_train.shape[1]} features")
    print(f"  Hidden layer: {nn_model.hidden_layer_sizes[0]} neurons (ReLU activation)")
    print(f"  Output layer: {len(split_data['genre_names'])} genres (softmax activation)")
    print(f"  Total parameters: ~{X_train.shape[1] * nn_model.hidden_layer_sizes[0] + nn_model.hidden_layer_sizes[0] * len(split_data['genre_names']):,}")
    
    return {
        'model': nn_model,
        'test_accuracy': test_accuracy,
        'feature_names': split_data['feature_names'],
        'genre_names': split_data['genre_names'],
        'confusion_matrix': cm
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
        print("  - Neural network training for music genre prediction")
        
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