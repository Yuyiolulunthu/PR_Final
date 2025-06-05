#!/usr/bin/env python3
"""
Complete Facial Expression Recognition Implementation using Classical Statistical Methods
Authors: Meng-Ling Tsai, Yu-Yi Chang
Course: 113-2 Pattern Recognition
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Data loading and preprocessing"""
    
    def __init__(self):
        self.emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 
                              'anger', 'disgust', 'fear', 'contempt']
    
    def load_fer_plus_data(self, csv_path):
        """Load FER+ format CSV data"""
        print("Loading FER+ data...")
        df = pd.read_csv(csv_path)
        
        images = []
        labels = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx} samples...")
            
            # Parse pixels
            try:
                if pd.isna(row['pixels']) or row['pixels'] == '':
                    continue
                    
                pixels = np.array(row['pixels'].split(), dtype=np.uint8)
                image = pixels.reshape(48, 48)
                
                # Determine emotion label from FER+ format
                emotion_scores = [
                    row['neutral'], row['happiness'], row['surprise'], row['sadness'],
                    row['anger'], row['disgust'], row['fear'], row['contempt']
                ]
                
                # Handle NaN values
                emotion_scores = [score if not pd.isna(score) else 0 for score in emotion_scores]
                
                # Skip if all emotions are 0
                if sum(emotion_scores) == 0:
                    continue
                
                # Use majority emotion
                emotion_label = np.argmax(emotion_scores)
                
                images.append(image)
                labels.append(emotion_label)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        print(f"Loaded {len(images)} samples")
        return np.array(images), np.array(labels)
    
    def detect_faces_viola_jones(self, image):
        """Face detection using Viola-Jones"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image, 1.1, 4)
        return faces
    
    def align_face(self, image, face_coords):
        """Face alignment and resizing"""
        if len(face_coords) > 0:
            x, y, w, h = face_coords[0]
            face = image[y:y+h, x:x+w]
        else:
            face = image
        
        # Resize to standard size
        face_resized = cv2.resize(face, (48, 48))
        return face_resized
    
    def normalize_illumination(self, image):
        """Illumination normalization using CLAHE"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        normalized = clahe.apply(image)
        
        # Gaussian smoothing
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        return blurred
    
    def preprocess_images(self, images):
        """Complete preprocessing pipeline"""
        processed_images = []
        
        for i, img in enumerate(images):
            if i % 1000 == 0:
                print(f"Preprocessing {i}/{len(images)}...")
            
            # Face detection
            faces = self.detect_faces_viola_jones(img)
            
            # Face alignment
            aligned_face = self.align_face(img, faces)
            
            # Illumination normalization
            normalized_face = self.normalize_illumination(aligned_face)
            
            processed_images.append(normalized_face)
        
        return np.array(processed_images)


class FeatureExtractor:
    """Feature extraction using LBP, HOG, and Gabor wavelets"""
    
    def __init__(self):
        # LBP parameters
        self.lbp_radius_list = [1, 2, 3]
        self.lbp_n_points_list = [8, 16, 24]
        
        # HOG parameters
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (8, 8)
        self.hog_cells_per_block = (2, 2)
        
        # Gabor parameters
        self.gabor_frequencies = [0.1, 0.3, 0.5]
        self.gabor_angles = [0, 45, 90, 135]
    
    def extract_lbp_features(self, image):
        """Extract Enhanced Multi-scale LBP features"""
        features = []
        
        for radius in self.lbp_radius_list:
            for n_points in self.lbp_n_points_list:
                # Calculate LBP
                lbp = local_binary_pattern(image, n_points, radius, method='uniform')
                
                # Calculate histogram
                hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                     range=(0, n_points + 2), density=True)
                features.extend(hist)
        
        return np.array(features)
    
    def extract_hog_features(self, image):
        """Extract HOG features"""
        features = hog(image, 
                      orientations=self.hog_orientations,
                      pixels_per_cell=self.hog_pixels_per_cell,
                      cells_per_block=self.hog_cells_per_block,
                      visualize=False,
                      feature_vector=True)
        return features
    
    def extract_gabor_features(self, image):
        """Extract Gabor wavelet features"""
        features = []
        
        for freq in self.gabor_frequencies:
            for angle in np.deg2rad(self.gabor_angles):
                real, _ = gabor(image, frequency=freq, theta=angle)
                
                # Statistical features
                features.extend([
                    np.mean(real),
                    np.std(real),
                    np.max(real),
                    np.min(real)
                ])
        
        return np.array(features)
    
    def extract_all_features(self, images):
        """Extract all features and fuse them"""
        print("Extracting features...")
        all_features = []
        
        for i, image in enumerate(images):
            if i % 1000 == 0:
                print(f"Processing {i}/{len(images)}...")
            
            # Extract individual features
            lbp_features = self.extract_lbp_features(image)
            hog_features = self.extract_hog_features(image)
            gabor_features = self.extract_gabor_features(image)
            
            # Normalize each feature type
            lbp_norm = (lbp_features - np.mean(lbp_features)) / (np.std(lbp_features) + 1e-8)
            hog_norm = (hog_features - np.mean(hog_features)) / (np.std(hog_features) + 1e-8)
            gabor_norm = (gabor_features - np.mean(gabor_features)) / (np.std(gabor_features) + 1e-8)
            
            # Concatenate all features
            combined_features = np.concatenate([lbp_norm, hog_norm, gabor_norm])
            all_features.append(combined_features)
        
        return np.array(all_features)
    
    def extract_individual_features(self, images):
        """Extract individual feature types for comparison"""
        lbp_features = []
        hog_features = []
        gabor_features = []
        
        print("Extracting individual features...")
        for i, image in enumerate(images):
            if i % 1000 == 0:
                print(f"Processing {i}/{len(images)}...")
            
            lbp_features.append(self.extract_lbp_features(image))
            hog_features.append(self.extract_hog_features(image))
            gabor_features.append(self.extract_gabor_features(image))
        
        return {
            'lbp': np.array(lbp_features),
            'hog': np.array(hog_features),
            'gabor': np.array(gabor_features)
        }


class FeatureSelector:
    """Feature selection using PCA and LDA"""
    
    def __init__(self):
        self.pca = None
        self.lda = None
        self.scaler = StandardScaler()
    
    def apply_pca(self, features, n_components=0.95):
        """Apply PCA for dimensionality reduction"""
        print("Applying PCA...")
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA
        self.pca = PCA(n_components=n_components)
        reduced_features = self.pca.fit_transform(features_scaled)
        
        print(f"PCA: {features.shape[1]} -> {reduced_features.shape[1]} features")
        print(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.3f}")
        
        return reduced_features
    
    def apply_lda(self, features, labels, n_components=6):
        """Apply LDA for further dimensionality reduction"""
        print("Applying LDA...")
        
        # Apply LDA
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        reduced_features = self.lda.fit_transform(features, labels)
        
        print(f"LDA: {features.shape[1]} -> {reduced_features.shape[1]} features")
        
        return reduced_features
    
    def transform_features(self, features):
        """Transform new features using fitted PCA and LDA"""
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        features_lda = self.lda.transform(features_pca)
        return features_lda


class EnsembleClassifier:
    """Ensemble classification using SVM, Random Forest, and Gradient Boosting"""
    
    def __init__(self):
        self.svm_model = None
        self.rf_model = None
        self.gb_model = None
        self.ensemble_model = None
        self.best_params = {}
    
    def train_svm(self, X_train, y_train):
        """Train SVM with hyperparameter tuning"""
        print("Training SVM...")
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
        
        svm = SVC(probability=True, random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, 
                                  scoring='f1_weighted', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        self.svm_model = grid_search.best_estimator_
        self.best_params['svm'] = grid_search.best_params_
        
        print(f"Best SVM parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return self.svm_model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest"""
        print("Training Random Forest...")
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        return self.rf_model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting"""
        print("Training Gradient Boosting...")
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
        self.gb_model.fit(X_train, y_train)
        
        return self.gb_model
    
    def create_ensemble(self, X_train, y_train):
        """Create and train ensemble model"""
        print("Creating ensemble model...")
        
        # Train individual models
        self.train_svm(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        # Create voting ensemble
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('svm', self.svm_model),
                ('rf', self.rf_model),
                ('gb', self.gb_model)
            ],
            voting='soft'
        )
        
        self.ensemble_model.fit(X_train, y_train)
        
        return self.ensemble_model
    
    def evaluate_individual_models(self, X_test, y_test):
        """Evaluate individual models"""
        results = {}
        
        models = {
            'SVM': self.svm_model,
            'Random Forest': self.rf_model,
            'Gradient Boosting': self.gb_model
        }
        
        for name, model in models.items():
            if model is not None:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = accuracy
                print(f"{name} Accuracy: {accuracy:.3f}")
        
        return results


class ModelEvaluator:
    """Model evaluation and visualization"""
    
    def __init__(self):
        self.emotion_labels = ['Neutral', 'Happy', 'Surprise', 'Sad', 
                              'Angry', 'Disgust', 'Fear', 'Contempt']
    
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Detailed report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.emotion_labels))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def plot_confusion_matrix(self, cm, title='Confusion Matrix'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.emotion_labels,
                   yticklabels=self.emotion_labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_comparison(self, results_dict):
        """Plot feature comparison results"""
        features = list(results_dict.keys())
        accuracies = list(results_dict.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(features, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        plt.title('Feature Type Comparison')
        plt.ylabel('Accuracy')
        plt.xlabel('Feature Type')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


class FERPipeline:
    """Complete FER pipeline"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.feature_selector = FeatureSelector()
        self.classifier = EnsembleClassifier()
        self.evaluator = ModelEvaluator()
        
        self.is_trained = False
    
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess data"""
        # Load data
        images, labels = self.preprocessor.load_fer_plus_data(csv_path)
        
        # Preprocess images
        processed_images = self.preprocessor.preprocess_images(images)
        
        return processed_images, labels
    
    def feature_analysis_experiment(self, images, labels):
        """Experiment with individual feature types"""
        print("\n" + "="*50)
        print("FEATURE ANALYSIS EXPERIMENT")
        print("="*50)
        
        # Extract individual features
        individual_features = self.feature_extractor.extract_individual_features(images)
        
        # Split data
        results = {}
        
        for feature_name, features in individual_features.items():
            print(f"\nTesting {feature_name.upper()} features...")
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Train simple SVM
            svm = SVC(kernel='rbf', random_state=42)
            svm.fit(X_train, y_train)
            
            # Evaluate
            y_pred = svm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[feature_name] = accuracy
            
            print(f"{feature_name.upper()} Accuracy: {accuracy:.3f}")
        
        # Plot comparison
        self.evaluator.plot_feature_comparison(results)
        
        return results
    
    def train_pipeline(self, images, labels):
        """Train complete pipeline"""
        print("\n" + "="*50)
        print("TRAINING COMPLETE PIPELINE")
        print("="*50)
        
        # Extract all features
        features = self.feature_extractor.extract_all_features(images)
        print(f"Original feature dimension: {features.shape[1]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Feature selection
        print("\nApplying feature selection...")
        X_train_pca = self.feature_selector.apply_pca(X_train)
        X_train_final = self.feature_selector.apply_lda(X_train_pca, y_train)
        
        # Transform test data
        X_test_final = self.feature_selector.transform_features(X_test)
        
        print(f"Final feature dimension: {X_train_final.shape[1]}")
        
        # Train ensemble classifier
        print("\nTraining ensemble classifier...")
        self.classifier.create_ensemble(X_train_final, y_train)
        
        # Evaluate individual models
        print("\nEvaluating individual models:")
        individual_results = self.classifier.evaluate_individual_models(X_test_final, y_test)
        
        # Evaluate ensemble
        print("\nEvaluating ensemble model:")
        ensemble_results = self.evaluator.evaluate_model(
            self.classifier.ensemble_model, X_test_final, y_test
        )
        
        # Plot confusion matrix
        self.evaluator.plot_confusion_matrix(
            ensemble_results['confusion_matrix'],
            'Ensemble Model Confusion Matrix'
        )
        
        self.is_trained = True
        
        return {
            'individual_results': individual_results,
            'ensemble_results': ensemble_results,
            'test_data': (X_test_final, y_test)
        }
    
    def predict_single_image(self, image):
        """Predict emotion for a single image"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Preprocess
        processed_image = self.preprocessor.normalize_illumination(image)
        
        # Extract features
        features = self.feature_extractor.extract_all_features([processed_image])
        
        # Transform features
        features_final = self.feature_selector.transform_features(features)
        
        # Predict
        prediction = self.classifier.ensemble_model.predict(features_final)[0]
        confidence = self.classifier.ensemble_model.predict_proba(features_final)[0]
        
        emotion_name = self.evaluator.emotion_labels[prediction]
        max_confidence = max(confidence)
        
        return emotion_name, max_confidence
    
    def cross_dataset_validation(self, train_data, test_data):
        """Cross-dataset validation"""
        print("\n" + "="*50)
        print("CROSS-DATASET VALIDATION")
        print("="*50)
        
        train_images, train_labels = train_data
        test_images, test_labels = test_data
        
        # Extract features
        train_features = self.feature_extractor.extract_all_features(train_images)
        test_features = self.feature_extractor.extract_all_features(test_images)
        
        # Feature selection on training data
        train_features_pca = self.feature_selector.apply_pca(train_features)
        train_features_final = self.feature_selector.apply_lda(train_features_pca, train_labels)
        
        # Transform test features
        test_features_final = self.feature_selector.transform_features(test_features)
        
        # Train on train dataset
        self.classifier.create_ensemble(train_features_final, train_labels)
        
        # Evaluate on test dataset
        results = self.evaluator.evaluate_model(
            self.classifier.ensemble_model, test_features_final, test_labels
        )
        
        return results


def main():
    """Main execution function"""
    print("Facial Expression Recognition using Classical Statistical Methods")
    print("="*70)
    
    # Initialize pipeline
    fer_pipeline = FERPipeline()
    
    # Load and preprocess data
    csv_path = "fer_plus_data.csv"  # Replace with your CSV path
    try:
        images, labels = fer_pipeline.load_and_preprocess_data(csv_path)
        print(f"Loaded {len(images)} samples with {len(np.unique(labels))} emotion classes")
        
        # Feature analysis experiment
        feature_results = fer_pipeline.feature_analysis_experiment(images, labels)
        
        # Train complete pipeline
        training_results = fer_pipeline.train_pipeline(images, labels)
        
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY")
        print("="*50)
        print(f"Ensemble Accuracy: {training_results['ensemble_results']['accuracy']:.3f}")
        print(f"Ensemble F1-Score: {training_results['ensemble_results']['f1_score']:.3f}")
        
        # Example prediction
        print("\nTesting single image prediction...")
        if len(images) > 0:
            emotion, confidence = fer_pipeline.predict_single_image(images[0])
            print(f"Predicted emotion: {emotion} (confidence: {confidence:.3f})")
        
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {csv_path}")
        print("Please ensure the CSV file path is correct.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your data format and file path.")


if __name__ == "__main__":
    main()