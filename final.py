#!/usr/bin/env python3
"""
Visual Agnosia Assistant for Raspberry Pi

This program helps individuals with visual agnosia by:
1. Detecting objects in real-time using YOLOv8
2. Speaking the names of detected objects using text-to-speech
3. Responding to voice commands like "What is in front of me?"

Requirements:
- Raspberry Pi 5 with camera module
- Python 3.7+
- PyTorch (lightweight version for Raspberry Pi)
- OpenCV
- Ultralytics YOLOv8
- pyttsx3 for text-to-speech
- Optional: SpeechRecognition for voice commands
"""

import os
import time
import threading
import queue
import cv2
import numpy as np
import torch
import pyttsx3

# Optional speech recognition
try:
    import speech_recognition as sr

    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    print("Speech Recognition not available. Voice commands will be disabled.")
    SPEECH_RECOGNITION_AVAILABLE = False

# Configuration
MODEL_PATH = "yolov8n.pt"  # Using YOLOv8 nano model for good Pi 5 performance
CONFIDENCE_THRESHOLD = 0.5
SPEECH_PAUSE_DURATION = 5  # Seconds between spoken announcements
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 20  # Pi 5 can handle higher FPS


class VisualAgnosiaAssistant:
    def __init__(self):
        print("Initializing Visual Agnosia Assistant...")

        # Initialize object detection model
        self.initialize_model()

        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Setting a comfortable speaking rate

        # Queue for TTS messages to avoid overlapping speech
        self.speech_queue = queue.Queue()
        self.last_speech_time = 0

        # Initialize speech recognition (if available)
        self.recognizer = None
        self.microphone = None
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            # Adjust for ambient noise
            with self.microphone as source:
                print("Calibrating for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source)

        # Initialize camera
        self.initialize_camera()

        # Track detected objects to minimize repetitive announcements
        self.recent_detections = set()
        self.last_detection_reset = time.time()

        # Start the speech output thread
        self.speech_thread = threading.Thread(target=self.speech_worker, daemon=True)
        self.speech_thread.start()

        # Start the speech recognition thread if available
        if SPEECH_RECOGNITION_AVAILABLE:
            self.recognition_thread = threading.Thread(target=self.listen_for_commands, daemon=True)
            self.recognition_thread.start()

        print("Initialization complete!")

    def initialize_model(self):
        """Initialize the YOLOv8 object detection model"""
        print("Loading YOLOv8 model...")
        try:
            from ultralytics import YOLO

            # Check if model file exists and is valid
            model_valid = False
            if os.path.exists(MODEL_PATH):
                try:
                    # Try loading the existing model file
                    self.model = YOLO(MODEL_PATH)
                    model_valid = True
                    print(f"Successfully loaded existing model from {MODEL_PATH}")
                except Exception as model_err:
                    print(f"Error loading existing model: {model_err}")
                    print("Will download a fresh model instead")
                    # If existing file is corrupt, remove it
                    if os.path.exists(MODEL_PATH):
                        try:
                            os.remove(MODEL_PATH)
                            print(f"Removed corrupted model file: {MODEL_PATH}")
                        except Exception as e:
                            print(f"Error removing corrupted model file: {e}")

            # If no valid model was loaded, download a fresh one
            if not model_valid:
                print("Downloading YOLOv8n model...")
                self.model = YOLO('yolov8n.pt')
                # Save the model for future use
                self.model.save(MODEL_PATH)
                print(f"Model downloaded and saved to {MODEL_PATH}")

            # GPU acceleration if available, otherwise use CPU
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {self.device}")

            print(f"YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"Error initializing model: {e}")
            self.speak("Error loading object detection model. Please check installation.")
            raise

    def initialize_camera(self):
        """Initialize the Raspberry Pi camera"""
        print("Initializing camera...")
        self.camera = cv2.VideoCapture(0)  # Use default camera (Pi camera)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        # Check if camera opened successfully
        if not self.camera.isOpened():
            print("Error: Could not open camera.")
            self.speak("Error: Camera not detected. Please check connection.")
            raise Exception("Camera initialization failed")

        print("Camera initialized successfully")

    def detect_objects(self, frame):
        """Detect objects in the given frame using YOLOv8"""
        # Perform detection directly on the frame (YOLOv8 handles BGR/RGB conversion)
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD)

        # Process results into a structured format
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                name = result.names[class_id]

                detections.append({
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2,
                    'confidence': confidence,
                    'name': name
                })

        return detections

    def process_detections(self, detections):
        """Process detected objects and queue them for speech output"""
        current_time = time.time()

        # Reset recent detections periodically to announce objects again
        if current_time - self.last_detection_reset > 30:  # Reset every 30 seconds
            self.recent_detections.clear()
            self.last_detection_reset = current_time

        # Collect new objects detected with confidence above threshold
        new_objects = set()
        for detection in detections:
            obj_name = detection['name']
            if obj_name not in self.recent_detections:
                new_objects.add(obj_name)
                self.recent_detections.add(obj_name)

        # Prepare speech about detected objects
        if new_objects and current_time - self.last_speech_time > SPEECH_PAUSE_DURATION:
            if len(new_objects) == 1:
                obj = list(new_objects)[0]
                self.queue_speech(f"I see a {obj}")
            elif len(new_objects) > 1:
                obj_list = ', '.join(list(new_objects)[:-1]) + ' and ' + list(new_objects)[-1]
                self.queue_speech(f"I see a {obj_list}")

            self.last_speech_time = current_time

    def speak(self, text):
        """Directly speak the given text (for immediate responses)"""
        print(f"Speaking: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def queue_speech(self, text):
        """Queue text to be spoken to prevent overlapping speech"""
        self.speech_queue.put(text)

    def speech_worker(self):
        """Worker thread that processes the speech queue"""
        while True:
            try:
                text = self.speech_queue.get(timeout=0.5)
                print(f"Speaking (from queue): {text}")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                pass  # No speech in queue, continue
            except Exception as e:
                print(f"Error in speech worker: {e}")
                time.sleep(1)  # Prevent tight loop on error

    def listen_for_commands(self):
        """Listen for voice commands in a separate thread"""
        if not SPEECH_RECOGNITION_AVAILABLE:
            print("Speech recognition is not available. Skipping command listening.")
            return

        print("Starting speech recognition...")

        while True:
            try:
                with self.microphone as source:
                    print("Listening for commands...")
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

                try:
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"Command recognized: {command}")

                    if "what" in command and "front" in command:
                        self.describe_current_view()
                    elif "hello" in command or "hi" in command:
                        self.queue_speech("Hello! I'm here to help you identify objects.")
                    elif "thank" in command:
                        self.queue_speech("You're welcome!")

                except sr.UnknownValueError:
                    pass  # Speech wasn't understood
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")

            except Exception as e:
                print(f"Error in voice command listener: {e}")
                time.sleep(1)  # Prevent tight loop on error

    def describe_current_view(self):
        """Describe what's currently visible in front of the camera"""
        # Capture current frame
        ret, frame = self.camera.read()
        if not ret:
            self.queue_speech("I'm having trouble seeing right now.")
            return

        # Detect objects
        detections = self.detect_objects(frame)

        # Describe what's visible
        if len(detections) == 0:
            self.queue_speech("I don't see any recognizable objects in front of you right now.")
        else:
            objects = set([detection['name'] for detection in detections])
            if len(objects) == 1:
                obj = list(objects)[0]
                self.queue_speech(f"There is a {obj} in front of you.")
            else:
                obj_list = ', '.join(list(objects)[:-1]) + ' and ' + list(objects)[-1]
                self.queue_speech(f"In front of you, I can see: {obj_list}.")

    def run(self):
        """Main processing loop"""
        print("Starting main processing loop...")
        self.queue_speech("Visual Agnosia Assistant is ready")

        try:
            while True:
                # Capture frame from camera
                ret, frame = self.camera.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    time.sleep(0.1)
                    continue

                # Process frame at a reduced rate to avoid overloading the Pi
                if int(time.time() * 2) % 2 == 0:  # Process every ~0.5 seconds
                    # Detect objects
                    detections = self.detect_objects(frame)

                    # Process and announce detections
                    self.process_detections(detections)

                    # Display results (optional, can be disabled for headless operation)
                    if 'DISPLAY' in os.environ:
                        # Draw detection boxes
                        for detection in detections:
                            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(
                                detection['xmax']), int(detection['ymax'])
                            label = f"{detection['name']} {detection['confidence']:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Display the frame
                        cv2.imshow('Visual Agnosia Assistant', frame)

                # Check for keyboard interrupt (press 'q' to quit)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Keyboard interrupt received. Shutting down...")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            # Cleanup
            self.camera.release()
            cv2.destroyAllWindows()
            print("Visual Agnosia Assistant shutdown complete")


if __name__ == "__main__":
    try:
        print("Starting Visual Agnosia Assistant with YOLOv8 on Raspberry Pi 5")
        assistant = VisualAgnosiaAssistant()
        assistant.run()
    except Exception as e:
        print(f"Fatal error: {e}")