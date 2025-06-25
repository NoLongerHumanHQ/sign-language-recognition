# Sign Language Recognition System

A real-time American Sign Language (ASL) recognition system built with MediaPipe, OpenCV, and Streamlit. This application uses machine learning to translate hand gestures from ASL alphabet (A-Z) into text.

## Features

- **Real-time hand gesture recognition** using webcam
- **American Sign Language (ASL) alphabet support** (A-Z)
- **Interactive data collection** interface
- **Model training** capabilities
- **Confidence scoring** for predictions
- **Clean, user-friendly interface** built with Streamlit

## Demo

![Sign Language Recognition Demo](https://i.imgur.com/demo.gif)

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sign-language-recognition.git
   cd sign-language-recognition
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

## Usage Guide

### Data Collection

1. Navigate to the "Data Collection" tab
2. Select which ASL signs you want to collect data for
3. Position your hand in the camera view
4. Click "Start Data Collection"
5. Follow the on-screen instructions to collect samples for each sign

### Model Training

1. Navigate to the "Training" tab
2. Select your collected data file
3. Configure model parameters (or use defaults)
4. Click "Train Model"
5. Wait for training to complete
6. Review model performance

### Recognition

1. Navigate to the "Recognition" tab
2. Select your trained model
3. Click "Start Recognition"
4. Position your hand in the camera view to see predictions

## How It Works

1. **Hand Detection**: MediaPipe Hands is used to detect and extract hand landmarks from the video feed
2. **Feature Extraction**: 21 hand landmarks with 3D coordinates (x, y, z) are extracted and normalized
3. **Classification**: A Random Forest classifier predicts the ASL sign based on the landmarks
4. **Output**: The predicted sign is displayed with a confidence score

## Project Structure

```
sign_language_recognition/
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── data_collection.py     # Data collection and preprocessing
├── utils.py              # Utility functions
├── requirements.txt      # Dependencies
├── README.md            # Project documentation
├── models/              # Trained model files
│   └── gesture_model.pkl
├── data/               # Dataset storage
│   ├── landmarks/      # Processed landmark data
│   └── raw/           # Raw gesture data
└── notebooks/         # Jupyter notebooks for experimentation
    └── model_training.ipynb
```

## Technical Details

### Data Processing

- Hand landmarks are extracted using MediaPipe Hands
- 21 landmarks (63 features) are normalized relative to the wrist position
- Data augmentation techniques include rotation, scaling, and noise addition

### Model Architecture

- Random Forest Classifier with optimized hyperparameters
- Input: 63 features (21 landmarks × 3 coordinates)
- Output: 26 classes (A-Z letters)

### Performance Optimizations

- Frame skipping for smoother video processing
- Model caching with `@st.cache_resource`
- Confidence thresholding to filter uncertain predictions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for hand landmark detection
- [Streamlit](https://streamlit.io/) for the interactive web interface 