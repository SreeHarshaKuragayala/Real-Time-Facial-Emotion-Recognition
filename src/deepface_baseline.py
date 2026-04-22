import cv2
from deepface import DeepFace
import time

cap = cv2.VideoCapture(0)

print("Starting emotion detection... Press Ctrl+C to stop")
print("=" * 50)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    try:
        # Analyze every 10th frame to improve performance
        if frame_count % 10 == 0:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotions = result[0]['emotion']
            dominant_emotion = max(emotions, key=emotions.get)

            # Clear previous output and display results
            print(f"\rDominant Emotion: {dominant_emotion} ({emotions[dominant_emotion]:.1f}%)", end="")

            # Show all emotions
            if frame_count % 50 == 0:  # Show detailed results every 50 frames
                print("\nAll emotions:")
                for emotion, confidence in emotions.items():
                    print(f"  {emotion}: {confidence:.1f}%")
                print("-" * 30)

        frame_count += 1

    except Exception as e:
        print(f"\nError: {e}")

    # Small delay to prevent excessive CPU usage
    time.sleep(0.1)

    # Break on keyboard interrupt
    try:
        pass
    except KeyboardInterrupt:
        print("\nStopping...")
        break

cap.release()
print("Camera released successfully")