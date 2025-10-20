"""
FMA Dataset Loader for Machine Learning

This module provides functionality to load the first N tracks' features and 
corresponding genre classifications from the Free Music Archive (FMA) dataset.
The loaded data is prepared for machine learning tasks but does not include
the actual ML implementation.

Author: Rowan Rosenberg
Date: October 2025
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.utils import shuffle
import utils


class FMADataLoader:
    """
    A class to load and preprocess FMA dataset for machine learning.
    
    This class handles loading tracks metadata, audio features, and genre information,
    and provides methods to extract the first N tracks with their features and labels.
    """
    
    def __init__(self, metadata_path='fma_metadata'):
        """
        Initialize the FMA data loader.
        
        Args:
            metadata_path (str): Path to the directory containing FMA metadata files
        """
        self.metadata_path = metadata_path
        self.tracks = None
        self.genres = None
        self.features = None
        self.echonest = None
        
    def load_metadata(self):
        """
        Load all metadata files from the FMA dataset.
        
        Returns:
            dict: Dictionary containing loaded dataframes with keys:
                  'tracks', 'genres', 'features', 'echonest'
        """
        print("Loading FMA metadata files...")
        
        # Construct full paths
        tracks_path = os.path.join(self.metadata_path, 'tracks.csv')
        genres_path = os.path.join(self.metadata_path, 'genres.csv')
        features_path = os.path.join(self.metadata_path, 'features.csv')
        echonest_path = os.path.join(self.metadata_path, 'echonest.csv')
        
        # Load data using the FMA utils module
        self.tracks = utils.load(tracks_path)
        self.genres = utils.load(genres_path)
        self.features = utils.load(features_path)
        self.echonest = utils.load(echonest_path)
        
        # Verify data integrity
        np.testing.assert_array_equal(self.features.index, self.tracks.index)
        assert self.echonest.index.isin(self.tracks.index).all()
        
        print(f"Loaded data shapes:")
        print(f"  Tracks: {self.tracks.shape}")
        print(f"  Genres: {self.genres.shape}")
        print(f"  Features: {self.features.shape}")
        print(f"  Echonest: {self.echonest.shape}")
        
        return {
            'tracks': self.tracks,
            'genres': self.genres,
            'features': self.features,
            'echonest': self.echonest
        }
    
    def get_first_n_tracks(self, n=1000, subset='small', feature_columns=None, 
                          multi_label=False, include_echonest=False):
        """
        Load the first N tracks with their features and genre classifications.
        
        Args:
            n (int): Number of tracks to load (default: 1000)
            subset (str): Dataset subset to use ('small', 'medium', 'large')
            feature_columns (list or str): Specific feature columns to extract.
                                          If None, uses all available features.
                                          Can be a single feature name or list of names.
            multi_label (bool): If True, return multiple genres per track.
                               If False, return only top-level genre (default: False)
            include_echonest (bool): Whether to include Echonest features (default: False)
        
        Returns:
            dict: Dictionary containing:
                - 'X': Feature matrix (numpy array)
                - 'y': Genre labels (numpy array)
                - 'track_ids': Track IDs (list)
                - 'feature_names': Feature column names (list)
                - 'genre_names': Genre names/classes (list)
                - 'metadata': Additional track metadata (DataFrame)
        """
        if self.tracks is None:
            self.load_metadata()
            
        print(f"Extracting first {n} tracks from '{subset}' subset...")
        
        # Filter by subset
        subset_mask = self.tracks['set', 'subset'] <= subset
        subset_tracks = self.tracks[subset_mask]
        subset_features = self.features[subset_mask]
        
        print(f"Available tracks in '{subset}' subset: {subset_tracks.shape[0]}")
        
        # Take first N tracks
        if n > subset_tracks.shape[0]:
            print(f"Warning: Requested {n} tracks but only {subset_tracks.shape[0]} available.")
            n = subset_tracks.shape[0]
        
        # Get first N track indices (they're already sorted by index)
        first_n_indices = subset_tracks.index[:n]
        
        # Extract tracks and features for first N
        selected_tracks = subset_tracks.loc[first_n_indices]
        selected_features = subset_features.loc[first_n_indices]
        
        # Handle feature selection
        if feature_columns is None:
            # Use all features
            if include_echonest:
                # Join with echonest features if requested
                echonest_subset = self.echonest[self.echonest.index.isin(first_n_indices)]
                X_features = selected_features.join(echonest_subset, how='inner')
                print(f"Including Echonest features. Reduced to {X_features.shape[0]} tracks with both feature types.")
                # Update selected tracks to match available echonest data
                selected_tracks = selected_tracks.loc[X_features.index]
                first_n_indices = X_features.index
            else:
                X_features = selected_features
        else:
            # Select specific feature columns
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]
            
            X_features = selected_features[feature_columns]
        
        # Extract features as numpy array
        X = X_features.values
        
        # Extract genre labels
        if multi_label:
            # Multiple genres per track
            genre_lists = selected_tracks['track', 'genres_all'].values
            # Use MultiLabelBinarizer to create binary matrix
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(genre_lists)
            genre_names = mlb.classes_.tolist()
            print(f"Multi-label classification: {len(genre_names)} genres")
        else:
            # Single top-level genre per track
            genre_series = selected_tracks['track', 'genre_top']
            # Remove any tracks without genre information
            valid_genre_mask = ~genre_series.isnull()
            
            if not valid_genre_mask.all():
                print(f"Removing {(~valid_genre_mask).sum()} tracks without genre information")
                genre_series = genre_series[valid_genre_mask]
                X = X[valid_genre_mask]
                first_n_indices = first_n_indices[valid_genre_mask]
                selected_tracks = selected_tracks[valid_genre_mask]
            
            # Encode genres as integers
            le = LabelEncoder()
            y = le.fit_transform(genre_series)
            genre_names = le.classes_.tolist()
            print(f"Single-label classification: {len(genre_names)} genres")
        
        # Get feature names
        if hasattr(X_features.columns, 'to_list'):
            feature_names = X_features.columns.to_list()
        else:
            # Handle MultiIndex columns
            feature_names = [str(col) for col in X_features.columns]
        
        # Extract additional metadata
        metadata_columns = [
            ('track', 'title'),
            ('track', 'duration'),
            ('track', 'listens'),
            ('album', 'title'),
            ('artist', 'name'),
        ]
        
        # Only include columns that exist
        existing_metadata_cols = []
        for col in metadata_columns:
            if col in selected_tracks.columns:
                existing_metadata_cols.append(col)
        
        metadata = selected_tracks[existing_metadata_cols] if existing_metadata_cols else None
        
        # Print summary
        print(f"\nDataset Summary:")
        print(f"  Number of tracks: {X.shape[0]}")
        print(f"  Number of features: {X.shape[1]}")
        print(f"  Number of genres: {len(genre_names)}")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Label array shape: {y.shape}")
        
        if len(genre_names) <= 20:  # Only print if not too many genres
            print(f"  Genres: {genre_names}")
        
        return {
            'X': X,
            'y': y,
            'track_ids': first_n_indices.tolist(),
            'feature_names': feature_names,
            'genre_names': genre_names,
            'metadata': metadata
        }
    
    def get_train_test_split(self, n=1000, subset='small', feature_columns=None,
                            multi_label=False, include_echonest=False, 
                            standardize=True, shuffle_data=True):
        """
        Load the first N tracks and return pre-split training and test sets.
        
        Args:
            n (int): Number of tracks to load
            subset (str): Dataset subset to use
            feature_columns: Specific feature columns to extract
            multi_label (bool): Multi-label classification
            include_echonest (bool): Include Echonest features
            standardize (bool): Whether to standardize features (default: True)
            shuffle_data (bool): Whether to shuffle training data (default: True)
        
        Returns:
            dict: Dictionary containing train/validation/test splits:
                - 'X_train', 'X_val', 'X_test': Feature matrices
                - 'y_train', 'y_val', 'y_test': Label arrays
                - 'feature_names': Feature names
                - 'genre_names': Genre names
                - 'scaler': StandardScaler object (if standardize=True)
        """
        # Load the data
        data = self.get_first_n_tracks(n=n, subset=subset, 
                                      feature_columns=feature_columns,
                                      multi_label=multi_label,
                                      include_echonest=include_echonest)
        
        # Get the indices for the selected tracks
        track_indices = pd.Index(data['track_ids'])
        
        # Create split masks based on existing train/validation/test splits
        train_mask = self.tracks.loc[track_indices, ('set', 'split')] == 'training'
        val_mask = self.tracks.loc[track_indices, ('set', 'split')] == 'validation'
        test_mask = self.tracks.loc[track_indices, ('set', 'split')] == 'test'
        
        # Split the data
        X = data['X']
        y = data['y']
        
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
        
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]
        
        # Shuffle training data
        if shuffle_data:
            X_train, y_train = shuffle(X_train, y_train, random_state=42)
        
        # Standardize features
        scaler = None
        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        print(f"\nTrain/Test Split Summary:")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Validation samples: {X_val.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': data['feature_names'],
            'genre_names': data['genre_names'],
            'scaler': scaler
        }


def main():
    """
    Example usage of the FMADataLoader.
    """
    print("FMA Dataset Loader - Example Usage")
    print("=" * 50)
    
    # Initialize loader with metadata path
    loader = FMADataLoader(metadata_path='fma_metadata')
    
    # Example 1: Load first 100 tracks with all features for single-label classification
    print("\nExample 1: First 100 tracks with all features (single-label)")
    data1 = loader.get_first_n_tracks(n=100, subset='small', multi_label=False)
    
    # Example 2: Load first 500 tracks with only MFCC features
    print("\nExample 2: First 500 tracks with MFCC features only")
    data2 = loader.get_first_n_tracks(n=500, subset='small', 
                                     feature_columns='mfcc', multi_label=False)
    
    # Example 3: Load with train/test split ready for ML
    print("\nExample 3: Train/test split for first 1000 tracks")
    split_data = loader.get_train_test_split(n=1000, subset='small', 
                                           feature_columns=['mfcc', 'spectral_contrast'],
                                           standardize=False)
    
    # Example 4: Multi-label classification
    print("\nExample 4: Multi-label classification with first 200 tracks")
    data4 = loader.get_first_n_tracks(n=200, subset='small', multi_label=True)
    
    print("\nData loading examples completed!")
    print("The data is now ready for machine learning algorithms.")


if __name__ == "__main__":
    main()