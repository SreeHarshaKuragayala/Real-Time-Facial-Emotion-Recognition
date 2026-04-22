import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
import time
import os
from collections import deque
import threading
import sys

# Import your model architecture (make sure this matches your training code)
from torchvision import models


def create_efficientnet_model(num_classes=8):
    """Create EfficientNet-B0 model - same as training"""
    model = models.efficientnet_b0(weights=None)  # No pretrained weights needed

    # Replace classifier to match training
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )

    return model


class LiveEmotionDetector:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")

        # Emotion classes (make sure this matches your training dataset)
        self.class_names = ['angry', 'contempt', 'disgusted', 'fearful', 'happiness', 'neutral', 'sadness', 'surprised']

        # Load model
        self.model = self.load_model(model_path)

        # Image preprocessing (same as training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Smoothing for stable predictions
        self.prediction_history = deque(maxlen=5)

        # FPS calculation
        self.fps_history = deque(maxlen=30)

        # Display settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2

    def load_model(self, model_path):
        """Load the trained model"""
        print(f"📥 Loading model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = create_efficientnet_model(num_classes=len(self.class_names))
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        print("✅ Model loaded successfully!")
        return model

    def preprocess_face(self, face_img):
        """Preprocess face image for prediction"""
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        face_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)

        return face_tensor

    def predict_emotion(self, face_tensor):
        """Predict emotion from face tensor"""
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()[0]

    def smooth_predictions(self, probabilities):
        """Apply temporal smoothing to predictions"""
        self.prediction_history.append(probabilities)

        if len(self.prediction_history) > 1:
            # Average recent predictions for stability
            smoothed = np.mean(list(self.prediction_history), axis=0)
            return smoothed

        return probabilities

    def format_predictions(self, probabilities):
        """Format predictions for display"""
        # Get top emotion
        top_emotion_idx = np.argmax(probabilities)
        top_emotion = self.class_names[top_emotion_idx]
        top_confidence = probabilities[top_emotion_idx] * 100

        # Format all emotions with percentages
        all_emotions = []
        for i, emotion in enumerate(self.class_names):
            confidence = probabilities[i] * 100
            all_emotions.append(f"{emotion}: {confidence:.1f}%")

        return top_emotion, top_confidence, all_emotions

    def draw_emotion_info(self, frame, x, y, w, h, top_emotion, top_confidence, probabilities):
        """Draw emotion information on frame"""
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw top emotion
        emotion_text = f"{top_emotion.upper()}: {top_confidence:.1f}%"
        cv2.putText(frame, emotion_text, (x, y - 10), self.font, self.font_scale, (0, 255, 0), self.thickness)

        # Draw emotion bar chart on the right side
        bar_x = frame.shape[1] - 250
        bar_y = 30
        bar_height = 20
        bar_spacing = 25

        # Background for emotion bars
        cv2.rectangle(frame, (bar_x - 10, bar_y - 10),
                      (frame.shape[1] - 10, bar_y + len(self.class_names) * bar_spacing + 10), (0, 0, 0), -1)

        # Draw emotion bars
        for i, (emotion, prob) in enumerate(zip(self.class_names, probabilities)):
            y_pos = bar_y + i * bar_spacing

            # Bar background
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + 200, y_pos + bar_height), (50, 50, 50), -1)

            # Bar fill
            bar_width = int(200 * prob)
            color = (0, 255, 0) if i == np.argmax(probabilities) else (255, 255, 255)
            cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_width, y_pos + bar_height), color, -1)

            # Text
            text = f"{emotion}: {prob * 100:.1f}%"
            cv2.putText(frame, text, (bar_x + 5, y_pos + 15), self.font, 0.5, (255, 255, 255), 1)

    def print_terminal_output(self, top_emotion, top_confidence, all_emotions, fps):
        """Print formatted output to terminal"""
        # Clear screen (works on most terminals)
        os.system('cls' if os.name == 'nt' else 'clear')

        print("🎭 LIVE EMOTION DETECTION")
        print("=" * 50)
        print(f"📹 FPS: {fps:.1f}")
        print(f"🎯 PRIMARY EMOTION: {top_emotion.upper()} ({top_confidence:.1f}%)")
        print("\n📊 ALL EMOTIONS:")
        print("-" * 30)

        for emotion_info in all_emotions:
            emotion, percentage = emotion_info.split(': ')
            # Add emoji for better visualization
            emoji_map = {
                'angry': '😠', 'contempt': '😤', 'disgusted': '🤢', 'fearful': '😨',
                'happiness': '😊', 'neutral': '😐', 'sadness': '😢', 'surprised': '😲'
            }
            emoji = emoji_map.get(emotion, '😐')
            print(f"{emoji} {emotion.capitalize()}: {percentage}")

        print("\n💡 Press 'q' to quit")
        print("=" * 50)

    def run(self):
        """Main detection loop"""
        print("🚀 Starting live emotion detection...")
        print("📷 Make sure your webcam is connected")
        print("💡 Press 'q' to quit")

        # Initialize camera
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("✅ Camera initialized successfully!")
        print("🎯 Looking for faces...")

        try:
            while True:
                start_time = time.time()

                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) > 0:
                    # Use the largest face
                    face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = face

                    # Extract face region
                    face_img = frame[y:y + h, x:x + w]

                    if face_img.size > 0:
                        # Preprocess face
                        face_tensor = self.preprocess_face(face_img)

                        # Predict emotion
                        probabilities = self.predict_emotion(face_tensor)

                        # Apply smoothing
                        smoothed_probabilities = self.smooth_predictions(probabilities)

                        # Format predictions
                        top_emotion, top_confidence, all_emotions = self.format_predictions(smoothed_probabilities)

                        # Draw on frame
                        self.draw_emotion_info(frame, x, y, w, h, top_emotion, top_confidence, smoothed_probabilities)

                        # Calculate FPS
                        fps = 1.0 / (time.time() - start_time)
                        self.fps_history.append(fps)
                        avg_fps = np.mean(list(self.fps_history))

                        # Print to terminal
                        self.print_terminal_output(top_emotion, top_confidence, all_emotions, avg_fps)

                else:
                    # No face detected
                    cv2.putText(frame, "No face detected", (50, 50), self.font, 1, (0, 0, 255), 2)

                # Display frame
                cv2.imshow('Live Emotion Detection', frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")

        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Camera released and windows closed")


def main():
    """Main function"""
    # Configuration
    MODEL_PATH = "../results/best_model_final.pth"  # Update this path if needed
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("🎭 Live Emotion Detection System")
    print("=" * 40)

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please make sure you have trained the model first!")
        return

    try:
        # Create detector
        detector = LiveEmotionDetector(MODEL_PATH, DEVICE)

        # Run detection
        detector.run()

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your model file and camera connection")


if __name__ == "__main__":
    main()