import cv2
import numpy as np
import os
import time
import argparse
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Dict, Optional

from utils import (
    init_mediapipe_hands,
    extract_hand_landmarks,
    preprocess_landmarks,
    augment_landmarks,
    save_landmarks_to_csv
)

# Define ASL alphabet letters
ASL_ALPHABET = [chr(i) for i in range(ord('A'), ord('Z')+1)]

class DataCollector:
    """
    Class for collecting hand gesture data.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        countdown_time: int = 3,
        capture_time: int = 2,
        samples_per_gesture: int = 20,
        camera_id: int = 0,
        augmentation_factor: int = 5
    ):
        """
        Initialize data collection parameters.
        
        Args:
            data_dir: Directory to store data
            countdown_time: Countdown time in seconds before capturing
            capture_time: Time in seconds to capture each gesture
            samples_per_gesture: Number of samples to capture per gesture
            camera_id: Camera ID to use (usually 0 for built-in webcam)
            augmentation_factor: Number of augmented samples to generate per real sample
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.landmarks_dir = os.path.join(data_dir, "landmarks")
        
        self.countdown_time = countdown_time
        self.capture_time = capture_time
        self.samples_per_gesture = samples_per_gesture
        self.camera_id = camera_id
        self.augmentation_factor = augmentation_factor
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.landmarks_dir, exist_ok=True)
        
        # Initialize MediaPipe hands
        self.hands = init_mediapipe_hands()
        
    def capture_gesture(
        self, 
        gesture_label: str, 
        display_callback=None
    ) -> List[np.ndarray]:
        """
        Capture samples for a single gesture.
        
        Args:
            gesture_label: Label for the gesture
            display_callback: Function to display the capture progress
            
        Returns:
            List of captured landmark arrays
        """
        cap = cv2.VideoCapture(self.camera_id)
        
        # Set camera properties for optimal performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        all_landmarks = []
        
        # Countdown phase
        start_time = time.time()
        while time.time() - start_time < self.countdown_time:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Resize for better performance
            frame = cv2.resize(frame, (640, 480))
            
            # Calculate time left in countdown
            time_left = self.countdown_time - int(time.time() - start_time)
            
            # Add countdown text
            cv2.putText(
                frame, 
                f"Get ready! {time_left}...", 
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            
            # Add gesture instruction
            cv2.putText(
                frame, 
                f"Show gesture: {gesture_label}", 
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Display if callback is provided
            if display_callback:
                display_callback(frame)
            else:
                cv2.imshow("Data Collection", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
        
        # Capture phase
        capture_start = time.time()
        frames_captured = 0
        
        while time.time() - capture_start < self.capture_time:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.resize(frame, (640, 480))
            
            # Extract hand landmarks
            landmarks, annotated_frame = extract_hand_landmarks(frame, self.hands)
            
            # If landmarks detected, collect them
            if landmarks is not None:
                all_landmarks.append(landmarks)
                frames_captured += 1
            
            # Calculate progress
            progress = min(100, int(100 * (time.time() - capture_start) / self.capture_time))
            
            # Add capture indication
            cv2.putText(
                annotated_frame, 
                f"Capturing: {progress}%", 
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 255), 
                2
            )
            
            cv2.putText(
                annotated_frame, 
                f"Frames captured: {frames_captured}", 
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Display if callback is provided
            if display_callback:
                display_callback(annotated_frame)
            else:
                cv2.imshow("Data Collection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break
        
        cap.release()
        if not display_callback:
            cv2.destroyAllWindows()
        
        return all_landmarks[:self.samples_per_gesture]  # Limit to required number of samples
    
    def collect_all_gestures(
        self, 
        gestures_to_collect: Optional[List[str]] = None,
        display_callback=None
    ) -> Dict[str, List[np.ndarray]]:
        """
        Collect data for all specified gestures.
        
        Args:
            gestures_to_collect: List of gesture labels to collect, defaults to ASL_ALPHABET
            display_callback: Function for displaying in Streamlit
            
        Returns:
            Dictionary mapping gestures to lists of landmark arrays
        """
        gestures_to_collect = gestures_to_collect or ASL_ALPHABET
        all_gesture_data = {}
        
        for gesture in gestures_to_collect:
            if display_callback:
                display_callback(f"Collecting data for gesture: {gesture}")
            else:
                print(f"Collecting data for gesture: {gesture}")
            
            landmarks = self.capture_gesture(gesture, display_callback)
            
            if landmarks:
                all_gesture_data[gesture] = landmarks
                
                # Generate augmented samples
                if self.augmentation_factor > 0:
                    augmented_samples = []
                    for lm in landmarks:
                        augmented_samples.extend(augment_landmarks(lm, num_augments=self.augmentation_factor))
                    
                    all_gesture_data[gesture].extend(augmented_samples)
                
                if display_callback:
                    display_callback(f"Collected {len(landmarks)} samples for gesture {gesture}, "
                                   f"plus {len(landmarks) * self.augmentation_factor} augmented samples")
                else:
                    print(f"Collected {len(landmarks)} samples for gesture {gesture}, "
                         f"plus {len(landmarks) * self.augmentation_factor} augmented samples")
            else:
                if display_callback:
                    display_callback(f"No valid samples collected for gesture {gesture}")
                else:
                    print(f"No valid samples collected for gesture {gesture}")
        
        return all_gesture_data
    
    def save_collected_data(self, collected_data: Dict[str, List[np.ndarray]]) -> str:
        """
        Save collected data to CSV file.
        
        Args:
            collected_data: Dictionary of collected landmark data
            
        Returns:
            Path to the saved CSV file
        """
        all_landmarks = []
        all_labels = []
        
        for gesture, landmarks_list in collected_data.items():
            all_landmarks.extend(landmarks_list)
            all_labels.extend([gesture] * len(landmarks_list))
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(self.landmarks_dir, f"landmarks_{timestamp}.csv")
        
        # Save landmarks to CSV
        save_landmarks_to_csv(all_landmarks, all_labels, csv_filename)
        
        return csv_filename

def run_data_collection_cli():
    """
    Run data collection from command line.
    """
    parser = argparse.ArgumentParser(description='Collect hand gesture data')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to store data')
    parser.add_argument('--countdown', type=int, default=3, help='Countdown time in seconds')
    parser.add_argument('--capture_time', type=int, default=2, help='Capture time per gesture in seconds')
    parser.add_argument('--samples', type=int, default=20, help='Samples per gesture')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--augmentation', type=int, default=5, help='Augmentation factor')
    parser.add_argument('--gestures', type=str, nargs='+', default=ASL_ALPHABET, help='Gestures to collect')
    
    args = parser.parse_args()
    
    collector = DataCollector(
        data_dir=args.data_dir,
        countdown_time=args.countdown,
        capture_time=args.capture_time,
        samples_per_gesture=args.samples,
        camera_id=args.camera,
        augmentation_factor=args.augmentation
    )
    
    collected_data = collector.collect_all_gestures(args.gestures)
    csv_path = collector.save_collected_data(collected_data)
    
    print(f"Data collection complete. Saved to: {csv_path}")
    print(f"Total samples collected: {sum(len(samples) for samples in collected_data.values())}")

def streamlit_data_collection_page():
    """
    Streamlit interface for data collection.
    """
    st.title("Sign Language Recognition - Data Collection")
    st.write("Collect hand gesture data for training the sign language recognition model.")
    
    # Sidebar configuration
    st.sidebar.header("Collection Settings")
    
    data_dir = st.sidebar.text_input("Data Directory", value="data")
    countdown_time = st.sidebar.slider("Countdown Time (seconds)", 1, 10, 3)
    capture_time = st.sidebar.slider("Capture Time (seconds)", 1, 10, 2)
    samples_per_gesture = st.sidebar.slider("Samples per Gesture", 5, 50, 20)
    augmentation_factor = st.sidebar.slider("Augmentation Factor", 0, 10, 5)
    
    # Select gestures to collect
    st.sidebar.header("Gestures")
    selected_gestures = []
    all_selected = st.sidebar.checkbox("Select All Gestures", True)
    
    if all_selected:
        selected_gestures = ASL_ALPHABET
    else:
        # Create multiple columns for better layout of checkboxes
        cols = st.sidebar.columns(3)
        for i, letter in enumerate(ASL_ALPHABET):
            if cols[i % 3].checkbox(letter, False):
                selected_gestures.append(letter)
    
    # Display current selection
    st.sidebar.write(f"Selected gestures: {', '.join(selected_gestures)}")
    
    # Main area for video display
    st.subheader("Camera Preview")
    video_placeholder = st.empty()
    status_text = st.empty()
    
    # Collection button
    if st.button("Start Data Collection"):
        if not selected_gestures:
            st.error("Please select at least one gesture to collect.")
            return
        
        collector = DataCollector(
            data_dir=data_dir,
            countdown_time=countdown_time,
            capture_time=capture_time,
            samples_per_gesture=samples_per_gesture,
            augmentation_factor=augmentation_factor
        )
        
        # Custom display function for Streamlit
        def display_frame(frame):
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            time.sleep(0.01)  # Small delay to allow UI update
        
        def update_status(text):
            status_text.info(text)
        
        # Run collection
        with st.spinner("Collecting gesture data..."):
            try:
                collected_data = collector.collect_all_gestures(
                    selected_gestures, 
                    display_callback=display_frame
                )
                
                if collected_data:
                    csv_path = collector.save_collected_data(collected_data)
                    total_samples = sum(len(samples) for samples in collected_data.values())
                    st.success(f"Data collection complete! Total samples: {total_samples}")
                    st.info(f"Data saved to: {csv_path}")
                else:
                    st.warning("No data was collected. Please try again.")
            except Exception as e:
                st.error(f"An error occurred during data collection: {str(e)}")
                raise e
    
    # Add instructions
    with st.expander("Instructions", expanded=False):
        st.markdown("""
        ### How to collect data
        
        1. Select gestures to collect from the sidebar
        2. Click "Start Data Collection"
        3. For each gesture:
           - Wait for the countdown
           - Show the requested sign clearly in the camera
           - Hold still during the capture phase
           - Repeat for all selected gestures
        
        ### Tips for better data quality
        
        - Use good lighting
        - Position your hand in the center of the frame
        - Use a plain background
        - Keep your hand at a consistent distance from the camera
        """)

if __name__ == "__main__":
    run_data_collection_cli()