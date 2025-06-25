import os
import numpy as np
import pandas as pd
import joblib
import argparse
import time
import streamlit as st
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from utils import (
    load_landmarks_from_csv,
    plot_confusion_matrix
)

# Default model hyperparameters
DEFAULT_RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'bootstrap': True,
    'class_weight': 'balanced'
}

class SignLanguageModel:
    """
    Class for training and evaluating sign language recognition models.
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        model_params: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model ('random_forest' or 'neural_network')
            model_params: Dictionary of model parameters
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type.lower()
        self.random_state = random_state
        
        if self.model_type == 'random_forest':
            self.model_params = model_params or DEFAULT_RF_PARAMS
            self.model = RandomForestClassifier(
                random_state=random_state,
                **self.model_params
            )
        elif self.model_type == 'neural_network':
            # For simplicity, we're not implementing the NN variant in detail
            # This would use TensorFlow/Keras and require more complex code
            raise NotImplementedError("Neural network model not implemented yet")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.label_encoder = LabelEncoder()
        
    def load_data(
        self,
        csv_file: str,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split data from a CSV file.
        
        Args:
            csv_file: Path to the CSV file containing landmark data
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X, y = load_landmarks_from_csv(csv_file)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state,
            stratify=y_encoded  # Ensure balanced classes in train/test
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Print feature importance if applicable
        if hasattr(self.model, 'feature_importances_'):
            top_indices = np.argsort(self.model.feature_importances_)[-10:]
            print("Top 10 features:")
            for i in top_indices:
                print(f"Feature {i}: {self.model.feature_importances_[i]:.4f}")
    
    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Optional[Dict] = None,
        cv: int = 5
    ) -> Dict:
        """
        Perform hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Grid of parameters to search
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of best parameters
        """
        if self.model_type == 'random_forest':
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=self.random_state),
                param_grid=param_grid,
                cv=cv,
                n_jobs=-1,
                scoring='accuracy',
                verbose=1
            )
            
            print("Starting hyperparameter tuning...")
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            tuning_time = time.time() - start_time
            
            print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
            
            # Update model with best parameters
            self.model = RandomForestClassifier(
                random_state=self.random_state,
                **grid_search.best_params_
            )
            
            return grid_search.best_params_
        else:
            raise NotImplementedError(f"Hyperparameter tuning not implemented for {self.model_type}")
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, Dict, np.ndarray]:
        """
        Evaluate the model.
        
        Args:
            X_test: Testing features
            y_test: Testing labels
            
        Returns:
            Tuple of (accuracy, classification_report_dict, confusion_matrix)
        """
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Test accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy, class_report, conf_matrix
        
    def save_model(self, model_dir: str = "models") -> str:
        """
        Save the trained model.
        
        Args:
            model_dir: Directory to save the model
            
        Returns:
            Path to the saved model file
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(model_dir, f"{self.model_type}_model_{timestamp}.pkl")
        
        # Save both the model and label encoder
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'classes': self.label_encoder.classes_
        }
        
        joblib.dump(model_data, model_filename)
        print(f"Model saved to {model_filename}")
        
        return model_filename
    
    @staticmethod
    def load_saved_model(model_path: str) -> Dict:
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Dictionary containing model and metadata
        """
        model_data = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        print(f"Classes: {model_data['classes']}")
        
        return model_data
    
    def predict(
        self,
        landmarks: np.ndarray,
        return_proba: bool = False
    ) -> Tuple[str, float]:
        """
        Predict the gesture for given landmarks.
        
        Args:
            landmarks: Preprocessed landmark features
            return_proba: Whether to return probability scores
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Reshape if needed
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(1, -1)
        
        # Predict class
        predicted_class_idx = self.model.predict(landmarks)[0]
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Get confidence
        if return_proba and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(landmarks)[0]
            confidence = proba[predicted_class_idx]
        else:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(landmarks)[0]
                confidence = proba[predicted_class_idx]
            else:
                # If model doesn't support probabilities, return fixed confidence
                confidence = 1.0
        
        return predicted_class, confidence

def run_training_cli():
    """
    Run model training from command line.
    """
    parser = argparse.ArgumentParser(description='Train sign language recognition model')
    parser.add_argument('--data_file', type=str, required=True, help='Path to landmarks CSV file')
    parser.add_argument('--model_type', type=str, default='random_forest', choices=['random_forest'],
                        help='Type of model to train')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save model')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set proportion')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_file}")
    trainer = SignLanguageModel(model_type=args.model_type)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = trainer.load_data(args.data_file, test_size=args.test_size)
    print(f"Data loaded. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Tune hyperparameters if requested
    if args.tune:
        print("Performing hyperparameter tuning...")
        trainer.tune_hyperparameters(X_train, y_train)
    
    # Train the model
    print(f"Training {args.model_type} model...")
    trainer.train(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating model...")
    accuracy, report, conf_matrix = trainer.evaluate(X_test, y_test)
    
    # Save the model
    model_path = trainer.save_model(args.model_dir)
    print(f"Training complete! Model saved to: {model_path}")
    print(f"Final accuracy: {accuracy:.4f}")

def streamlit_model_training_page():
    """
    Streamlit interface for model training.
    """
    st.title("Sign Language Recognition - Model Training")
    
    # Sidebar configurations
    st.sidebar.header("Training Settings")
    
    # Data file selection
    data_files = []
    landmarks_dir = os.path.join("data", "landmarks")
    if os.path.exists(landmarks_dir):
        data_files = [f for f in os.listdir(landmarks_dir) if f.endswith(".csv")]
    
    if not data_files:
        st.warning("No landmark data files found. Please collect data first.")
        
        # Add a link to the data collection page
        st.info("Go to the Data Collection page to collect landmark data.")
        return
    
    selected_data_file = st.sidebar.selectbox("Select data file", data_files)
    data_file_path = os.path.join(landmarks_dir, selected_data_file)
    
    # Model configuration
    model_type = st.sidebar.radio("Model Type", ["Random Forest"])
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        tune_hyperparams = st.checkbox("Tune Hyperparameters", False)
        
        if model_type == "Random Forest":
            n_estimators = st.number_input("Number of Trees", 10, 500, 100, 10)
            max_depth = st.number_input("Maximum Depth", 5, 50, 20, 5)
            
            model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'bootstrap': True,
                'class_weight': 'balanced'
            }
    
    # Display data information
    if st.checkbox("Show Data Information"):
        try:
            X, y = load_landmarks_from_csv(data_file_path)
            classes, counts = np.unique(y, return_counts=True)
            
            st.write(f"Total samples: {len(y)}")
            st.write(f"Number of classes: {len(classes)}")
            st.write(f"Classes: {', '.join(classes)}")
            
            # Display class distribution
            class_data = pd.DataFrame({
                'Class': classes,
                'Count': counts
            }).sort_values(by='Count', ascending=False)
            
            st.bar_chart(class_data.set_index('Class'))
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    # Train model button
    if st.button("Train Model"):
        try:
            # Initialize model
            trainer = SignLanguageModel(
                model_type='random_forest' if model_type == "Random Forest" else "neural_network",
                model_params=model_params
            )
            
            # Load data
            with st.spinner("Loading and preprocessing data..."):
                X_train, X_test, y_train, y_test = trainer.load_data(
                    data_file_path, 
                    test_size=test_size
                )
                
                st.info(f"Data loaded. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
            
            # Tune hyperparameters if requested
            if tune_hyperparams:
                with st.spinner("Tuning hyperparameters (this may take a while)..."):
                    best_params = trainer.tune_hyperparameters(X_train, y_train)
                    st.success("Hyperparameter tuning complete!")
                    st.json(best_params)
            
            # Train model
            with st.spinner("Training model..."):
                trainer.train(X_train, y_train)
                st.success("Model training complete!")
            
            # Evaluate model
            with st.spinner("Evaluating model..."):
                accuracy, report, conf_matrix = trainer.evaluate(X_test, y_test)
                
                st.subheader("Model Performance")
                st.metric("Test Accuracy", f"{accuracy:.2%}")
                
                # Display classification report
                st.subheader("Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.2f}"))
                
                # Display confusion matrix
                st.subheader("Confusion Matrix")
                fig = plot_confusion_matrix(
                    y_test,
                    trainer.model.predict(X_test),
                    trainer.label_encoder.classes_,
                    normalize=True
                )
                st.pyplot(fig)
            
            # Save model
            with st.spinner("Saving model..."):
                model_path = trainer.save_model()
                st.success(f"Model saved to {model_path}")
        
        except Exception as e:
            st.error(f"An error occurred during model training: {str(e)}")
            raise e

if __name__ == "__main__":
    run_training_cli()