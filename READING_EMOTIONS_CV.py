import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp


class EmotionDetector:
    def __init__(self, model_path: str, use_grayscale: bool = True, confidence_threshold: float = 0.7):
        self.use_grayscale = use_grayscale
        self.input_shape = (48, 48, 1) if use_grayscale else (48, 48, 3)
        self.label_map = {
            0: 'Angry',
            1: 'Happy',
            2: 'Sad',
            3: 'Surprise',
            4: 'Neutral'
        }

        # Load model
        self.model = load_model(model_path)

        # Initialize MediaPipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=confidence_threshold)

    def preprocess_face(self, face_img):
        try:
            if self.use_grayscale:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(face_img, (48, 48))
            normalized = resized_face / 255.0
            processed = normalized.reshape(1, *self.input_shape)
            return processed
        except Exception as e:
            print(f"[ERROR] Preprocessing error: {str(e)}")
            return None

    def predict_emotion(self, face_img):
        processed_face = self.preprocess_face(face_img)
        if processed_face is not None:
            prediction = self.model.predict(processed_face, verbose=0)[0]
            emotion_index = np.argmax(prediction)
            emotion_label = self.label_map.get(emotion_index, "Unknown")
            confidence = float(np.max(prediction))
            return emotion_label, confidence, prediction
        return None, None, None

    def detect_emotions(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        print("[INFO] Starting webcam... Press 'q' to exit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    x2 = min(x + width, w)
                    y2 = min(y + height, h)
                    face_roi = frame[y:y2, x:x2]

                    if face_roi.size > 0:
                        emotion_label, confidence, prediction = self.predict_emotion(face_roi)
                        if emotion_label:
                            # Draw bounding box and label
                            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{emotion_label} ({confidence:.2f})",
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 255, 0), 2)

            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quitting...")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Update the model path if needed
    MODEL_PATH = "saved_models/best_emotion_model.keras"
    detector = EmotionDetector(MODEL_PATH, use_grayscale=True)
    detector.detect_emotions()