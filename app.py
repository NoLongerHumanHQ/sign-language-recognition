import os
import cv2
import time
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Tuple, Dict, List
from collections import deque

from utils import (
    init_mediapipe_hands,
    extract_hand_landmarks,
    preprocess_landmarks,
    draw_text_prediction
)

from data_collection import streamlit_data_collection_page
from train_model import streamlit_model_training_page, SignLanguageModel

# Set page configuration
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables for model and MediaPipe
@st.cache_resource
def load_mediapipe_model():
    """
    Load MediaPipe Hands model with caching.
    """
    return init_mediapipe_hands(min_detection_confidence=0.7)

@st.cache_resource
def load_recognition_model(model_path: str):
    """
    Load sign language recognition model with caching.
    """
    if os.path.exists(model_path):
        try:
            model_data = SignLanguageModel.load_saved_model(model_path)
            return model_data
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    else:
        return None

def get_available_models() -> List[str]:
    """
    Get list of available trained models.
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    return model_files

def recognition_page():
    """
    Main page for real-time sign language recognition.
    """
    st.title("Sign Language Recognition")
    st.write("Recognize ASL hand gestures in real-time")
    
    # Sidebar for model selection and settings
    st.sidebar.header("Recognition Settings")
    
    # Model selection
    model_files = get_available_models()
    
    if not model_files:
        st.warning("No trained models found. Please go to the Training page to train a model.")
        return
    
    selected_model = st.sidebar.selectbox("Select model", model_files)
    model_path = os.path.join("models", selected_model)
    
    # Load the model
    model_data = load_recognition_model(model_path)
    
    if not model_data:
        st.error("Failed to load model. Please train a new model or select a different one.")
        return
    
    # Extract model components
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    classes = model_data['classes']
    
    # Display available classes
    with st.expander("Available Classes"):
        st.write(f"This model can recognize {len(classes)} classes: {', '.join(classes)}")
    
    # Recognition settings
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence score to display a prediction"
    )
    
    # Frame processing settings
    frame_skip = st.sidebar.slider(
        "Process every N frame",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        help="Higher values improve performance but reduce smoothness"
    )
    
    # History settings
    history_length = st.sidebar.slider(
        "Prediction history length",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )
    
    # Camera settings
    camera_id = st.sidebar.number_input("Camera ID", min_value=0, max_value=10, value=0, step=1)
    
    # Main area for video display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Feed")
        frame_placeholder = st.empty()
    
    with col2:
        st.subheader("Recognition Results")
        result_placeholder = st.empty()
        
        # Prediction history section
        st.subheader("Prediction History")
        history_placeholder = st.empty()
        
        # Confidence visualization
        confidence_placeholder = st.empty()
    
    # Start/Stop button
    start_button = st.button("Start Recognition", key="start_button")
    stop_button = st.button("Stop Recognition", key="stop_button")
    
    # Initialize prediction history
    prediction_history = deque(maxlen=history_length)
    frame_count = 0
    
    # Load MediaPipe model
    hands = load_mediapipe_model()
    
    if start_button:
        # Initialize webcam
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            st.error("Could not open webcam. Please check camera settings.")
            return
        
        st.info("Recognition started. Press 'Stop Recognition' to end the session.")
        
        while not stop_button:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                st.error("Failed to get frame from camera. Please try again.")
                break
            
            # Flip horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            # Process every Nth frame for better performance
            if frame_count % frame_skip == 0:
                # Extract hand landmarks
                landmarks, processed_frame = extract_hand_landmarks(frame, hands)
                
                # If hand landmarks detected
                if landmarks is not None:
                    # Preprocess landmarks for prediction
                    processed_landmarks = preprocess_landmarks(landmarks)
                    
                    # Make prediction
                    predicted_class, confidence = model.predict(processed_landmarks, return_proba=True)
                    
                    # Add to prediction history
                    prediction_history.append((predicted_class, confidence))
                    
                    # Draw result on frame if confidence exceeds threshold
                    if confidence >= confidence_threshold:
                        processed_frame = draw_text_prediction(
                            processed_frame,
                            predicted_class,
                            confidence
                        )
                        result_placeholder.markdown(f"<h1 style='text-align: center; font-size: 5em;'>{predicted_class}</h1>", unsafe_allow_html=True)
                    else:
                        result_placeholder.markdown("<h1 style='text-align: center; color: gray; font-size: 5em;'>?</h1>", unsafe_allow_html=True)
                else:
                    # No hand detected
                    processed_frame = frame
                    result_placeholder.markdown("<h1 style='text-align: center; color: gray; font-size: 5em;'>No hand detected</h1>", unsafe_allow_html=True)
                
                # Show the frame
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Update prediction history visualization
                if prediction_history:
                    # Create dataframe for visualization
                    history_data = []
                    for i, (pred_class, conf) in enumerate(prediction_history):
                        history_data.append({
                            "Timestamp": i,
                            "Class": pred_class,
                            "Confidence": conf
                        })
                    
                    history_df = pd.DataFrame(history_data)
                    
                    # Show last prediction confidence
                    if history_data:
                        last_prediction = history_data[-1]
                        confidence_fig = px.bar(
                            x=[last_prediction["Class"]],
                            y=[last_prediction["Confidence"]],
                            range_y=[0, 1],
                            labels={"x": "Class", "y": "Confidence"},
                            title="Current Prediction Confidence",
                            text_auto=True
                        )
                        confidence_placeholder.plotly_chart(confidence_fig, use_container_width=True)
                        
                        # Show history as a table
                        history_placeholder.dataframe(
                            history_df[["Class", "Confidence"]],
                            hide_index=True,
                            use_container_width=True
                        )
            
            # Increment frame counter
            frame_count += 1
            
            # Add small delay to prevent UI lag
            time.sleep(0.01)
        
        # Release resources
        cap.release()

def main():
    """
    Main function to run the Streamlit app.
    """
    # Add custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1em;
    }
    .info-text {
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("<h1 class='main-header'>Sign Language Recognition</h1>", unsafe_allow_html=True)
    app_mode = st.sidebar.selectbox(
        "Choose a mode",
        ["Recognition", "Data Collection", "Training", "About"]
    )
    
    # Display selected page
    if app_mode == "Recognition":
        recognition_page()
    elif app_mode == "Data Collection":
        streamlit_data_collection_page()
    elif app_mode == "Training":
        streamlit_model_training_page()
    elif app_mode == "About":
        about_page()

def about_page():
    """
    About page with project information.
    """
    st.title("About Sign Language Recognition")
    
    st.markdown("""
    ## Overview
    
    This application uses computer vision and machine learning to recognize American Sign Language (ASL) 
    hand gestures in real-time. It can recognize the ASL alphabet (A-Z) through a webcam.
    
    ## How It Works
    
    1. **Hand Detection**: We use MediaPipe Hands to detect hand landmarks in the video stream
    2. **Feature Extraction**: 21 hand landmarks (63 coordinates) are extracted and normalized
    3. **Classification**: A machine learning model (Random Forest) predicts the sign
    4. **Visualization**: Results are displayed with confidence scores
    
    ## Usage Instructions
    
    1. Start with the **Data Collection** page to gather training data
    2. Go to the **Training** page to train a model on your data
    3. Use the **Recognition** page to test the model in real-time
    
    ## Tips for Best Results
    
    - Use good lighting conditions
    - Position your hand clearly in the frame
    - Keep a consistent distance from the camera
    - Train with your own gestures for the best accuracy
    
    ## Project Structure
    
    - `app.py` - Main Streamlit application
    - `data_collection.py` - Tools for gathering training data
    - `train_model.py` - Model training pipeline
    - `utils.py` - Utility functions for processing landmarks
    """)
    
    # System information
    with st.expander("System Information"):
        st.write(f"- OpenCV version: {cv2.__version__}")
        st.write(f"- NumPy version: {np.__version__}")
        st.write("- MediaPipe Hands: v0.10.5")
        st.write(f"- Models directory: {os.path.abspath('models')}")
        st.write(f"- Data directory: {os.path.abspath('data')}")

if __name__ == "__main__":
    main()