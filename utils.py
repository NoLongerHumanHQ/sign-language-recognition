import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def init_mediapipe_hands(
    static_image_mode: bool = False,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5
) -> mp_hands.Hands:
    """
    Initialize the MediaPipe Hands solution.
    
    Args:
        static_image_mode: Whether to treat the input images as a batch of static images
        max_num_hands: Maximum number of hands to detect
        min_detection_confidence: Minimum confidence for hand detection
        min_tracking_confidence: Minimum confidence for hand tracking
        
    Returns:
        MediaPipe Hands solution instance
    """
    try:
        hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        logger.info("MediaPipe Hands initialized successfully")
        return hands
    except Exception as e:
        logger.error(f"Failed to initialize MediaPipe Hands: {str(e)}")
        raise

def extract_hand_landmarks(
    image: np.ndarray, 
    hands_model: mp_hands.Hands,
    draw: bool = True
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Extract hand landmarks from an image.
    
    Args:
        image: Input image (BGR format)
        hands_model: MediaPipe Hands model
        draw: Whether to draw landmarks on the image
        
    Returns:
        Tuple containing (landmarks_array, processed_image)
    """
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # To improve performance, mark the image as not writeable
    image_rgb.flags.writeable = False
    
    # Process the image and detect hands
    results = hands_model.process(image_rgb)
    
    # Make the image writeable again for drawing
    image_rgb.flags.writeable = True
    image_output = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    landmarks_array = None
    
    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Get the first hand
        
        if draw:
            mp_drawing.draw_landmarks(
                image_output,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Extract landmark coordinates
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    
    return landmarks_array, image_output

def preprocess_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Preprocess hand landmarks for model input.
    
    Args:
        landmarks: Numpy array of shape (21, 3) containing landmark coordinates
        
    Returns:
        Preprocessed landmark features as a flat array
    """
    if landmarks is None:
        return np.zeros(63)  # Return zeros if no landmarks
    
    # Normalize coordinates relative to wrist position
    wrist = landmarks[0]
    normalized_landmarks = landmarks - wrist
    
    # Flatten landmarks to a 1D array (21 landmarks * 3 coordinates = 63 features)
    flattened = normalized_landmarks.flatten()
    
    # Scale features to a reasonable range
    # Using min-max scaling to [0, 1]
    if np.max(np.abs(flattened)) > 0:
        flattened = flattened / np.max(np.abs(flattened))
    
    return flattened

def augment_landmarks(
    landmarks: np.ndarray,
    num_augments: int = 5,
    noise_level: float = 0.005,
    rotation_range: float = 0.1,
    scale_range: float = 0.1
) -> List[np.ndarray]:
    """
    Generate augmented versions of hand landmarks.
    
    Args:
        landmarks: Original landmarks array (21, 3)
        num_augments: Number of augmented samples to generate
        noise_level: Level of Gaussian noise to add
        rotation_range: Maximum rotation angle in radians
        scale_range: Maximum scaling factor
        
    Returns:
        List of augmented landmark arrays
    """
    if landmarks is None:
        return []
    
    augmented_samples = []
    
    for _ in range(num_augments):
        # Make a copy of the original landmarks
        aug_landmarks = landmarks.copy()
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, aug_landmarks.shape)
        aug_landmarks += noise
        
        # Apply random rotation (in 2D space - x,y coordinates)
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range)
            cos_val, sin_val = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_val, -sin_val],
                [sin_val, cos_val]
            ])
            aug_landmarks[:, 0:2] = aug_landmarks[:, 0:2] @ rotation_matrix
        
        # Apply random scaling
        if scale_range > 0:
            scale = np.random.uniform(1 - scale_range, 1 + scale_range)
            aug_landmarks *= scale
        
        augmented_samples.append(aug_landmarks)
    
    return augmented_samples

def draw_text_prediction(
    image: np.ndarray,
    text: str,
    confidence: float,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 1.0,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw prediction text with confidence on the image.
    
    Args:
        image: Input image
        text: Text to display
        confidence: Confidence score (0.0 to 1.0)
        position: Position of the text (x, y)
        font_scale: Font scale
        thickness: Text thickness
        
    Returns:
        Image with text overlay
    """
    image_with_text = image.copy()
    
    # Determine color based on confidence
    if confidence >= 0.8:
        color = (0, 255, 0)  # Green for high confidence
    elif confidence >= 0.5:
        color = (0, 255, 255)  # Yellow for medium confidence
    else:
        color = (0, 0, 255)  # Red for low confidence
    
    # Draw text with a black outline for better visibility
    cv2.putText(
        image_with_text,
        f"{text} ({confidence:.2f})",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2
    )
    
    cv2.putText(
        image_with_text,
        f"{text} ({confidence:.2f})",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    
    return image_with_text

def save_landmarks_to_csv(
    landmarks_list: List[np.ndarray],
    labels: List[str],
    filename: str
) -> None:
    """
    Save landmarks and their labels to a CSV file.
    
    Args:
        landmarks_list: List of landmark arrays
        labels: List of corresponding labels
        filename: Output CSV filename
    """
    if not landmarks_list:
        logger.warning("No landmarks to save")
        return
    
    # Prepare data for DataFrame
    data = []
    for i, landmarks in enumerate(landmarks_list):
        if landmarks is not None:
            flat_landmarks = landmarks.flatten()
            row = np.append(flat_landmarks, labels[i])
            data.append(row)
    
    if not data:
        logger.warning("No valid landmarks to save")
        return
    
    # Create column names
    columns = [f"landmark_{i}_{coord}" for i in range(21) for coord in ['x', 'y', 'z']]
    columns.append("label")
    
    # Create and save DataFrame
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    logger.info(f"Saved {len(df)} samples to {filename}")

def load_landmarks_from_csv(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load landmarks and labels from a CSV file.
    
    Args:
        filename: Input CSV filename
        
    Returns:
        Tuple of (landmarks_array, labels_array)
    """
    try:
        df = pd.read_csv(filename)
        
        # Extract labels
        labels = df['label'].values
        
        # Extract landmarks
        feature_columns = [col for col in df.columns if col.startswith('landmark_')]
        features = df[feature_columns].values
        
        logger.info(f"Loaded {len(df)} samples from {filename}")
        return features, labels
    except Exception as e:
        logger.error(f"Error loading landmarks from {filename}: {str(e)}")
        return np.array([]), np.array([])

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap: plt.cm = plt.cm.Blues
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        cmap: Color map
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )
    
    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    fig.tight_layout()
    return fig 