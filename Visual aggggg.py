#!/usr/bin/env python3
"""
Visual Agnosia Assistance Program - Laptop Version

This program uses a laptop webcam to detect objects in real-time and
provides audio feedback via text-to-speech.
"""

import os
import time
import threading
import queue
import numpy as np
import cv2
import pyttsx3
import torch

class VisualAssistant:
    def __init__(self, camera_id=0):
        self.setup_camera(camera_id)
        self.setup_tts()
        self.setup_yolo()
        
        # Queue for handling speech requests
        self.speech_queue = queue.Queue()
        self.last_detection_time = 0
        self.detection_cooldown = 3  # seconds between announcements
        self.running = True
        
        # Thread for handling speech
        self.speech_thread = threading.Thread(target=self.speech_worker)
        self.speech_thread.daemon = True
        self.speech_thread.start()
    
    def setup_camera(self, camera_id):
        """Initialize the laptop's webcam"""
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise ValueError("Could not open webcam. Please check if the camera is connected properly.")
        
        # Set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Camera initialized")
    
    def setup_tts(self):
        """Initialize text-to-speech engine"""
        self.engine = pyttsx3.init()
        # Adjust speech rate and volume
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        print("Text-to-speech initialized")
    
    def setup_yolo(self):
        """Initialize YOLO model for object detection"""
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Optimize for inference
        self.model.eval()
        
        # Use GPU if available
        if torch.cuda.is_available():
            self.model.cuda()
            print("Using GPU for YOLO inference")
        else:
            # Optimize model for CPU inference
            self.model = self.model.to(torch.device('cpu'))
            print("Using CPU for YOLO inference")
        
        print("YOLO model loaded")
    
    def capture_frame(self):
        """Capture a frame from the webcam"""
        ret, frame = self.camera.read()
        if not ret:
            print("Warning: Could not read frame from webcam")
            return None
        return frame
    
    def detect_objects(self, frame):
        """Detect objects in the frame using YOLO"""
        if frame is None:
            return []
        
        # Convert frame for YOLO processing
        results = self.model(frame)
        
        # Get detected objects
        detections = results.pandas().xyxy[0]  # Get detection results
        return detections
    
    def speak(self, text):
        """Add text to speech queue"""
        self.speech_queue.put(text)
    
    def speech_worker(self):
        """Worker thread to handle text-to-speech requests"""
        while self.running:
            try:
                text = self.speech_queue.get(timeout=1)
                print(f"Speaking: {text}")
                self.engine.say(text)
                self.engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                continue
    
    def announce_objects(self, force=False):
        """Announce detected objects"""
        current_time = time.time()
        
        # Only announce if forced or enough time has passed since last announcement
        if force or (current_time - self.last_detection_time) > self.detection_cooldown:
            frame = self.capture_frame()
            if frame is not None:
                detections = self.detect_objects(frame)
                
                if len(detections) > 0:
                    # Group objects by class
                    objects = {}
                    for _, row in detections.iterrows():
                        obj_name = row['name']
                        confidence = row['confidence']
                        
                        if confidence > 0.5:  # Only include objects with high confidence
                            if obj_name in objects:
                                objects[obj_name] += 1
                            else:
                                objects[obj_name] = 1
                    
                    # Create announcement text
                    if objects:
                        announcement = "I can see "
                        object_list = []
                        
                        for obj_name, count in objects.items():
                            if count > 1:
                                object_list.append(f"{count} {obj_name}s")
                            else:
                                object_list.append(f"a {obj_name}")
                        
                        if len(object_list) == 1:
                            announcement += object_list[0]
                        elif len(object_list) == 2:
                            announcement += f"{object_list[0]} and {object_list[1]}"
                        else:
                            announcement += ", ".join(object_list[:-1]) + f", and {object_list[-1]}"
                        
                        self.speak(announcement)
                    else:
                        self.speak("I don't see any recognizable objects.")
                else:
                    self.speak("I don't see any objects.")
                
                self.last_detection_time = current_time
    
    def display_frame(self, frame, detections):
        """Display frame with bounding boxes"""
        if frame is None:
            return
        
        # Create a copy of the frame to draw on
        display_frame = frame.copy()
        
        # Draw bounding boxes around detected objects
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            obj_name = row['name']
            confidence = row['confidence']
            
            if confidence > 0.5:  # Only display objects with high confidence
                # Draw rectangle
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{obj_name}: {confidence:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Visual Assistant", display_frame)
        
        # Add instructions to the window
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            self.announce_objects(force=True)
        elif key == ord('q'):
            self.running = False
    
    def run(self):
        """Main loop"""
        print("Starting Visual Assistant.")
        print("Press 's' to scan and announce objects")
        print("Press 'q' to quit")
        
        self.speak("Visual assistant is ready.")
        
        try:
            # Initialize variables for FPS calculation
            start_time = time.time()
            frame_count = 0
            
            while self.running:
                # Capture a frame
                frame = self.capture_frame()
                if frame is None:
                    continue
                
                # Process every 3rd frame to balance performance
                if frame_count % 3 == 0:
                    # Detect objects
                    detections = self.detect_objects(frame)
                    
                    # Show the video feed with bounding boxes
                    self.display_frame(frame, detections)
                    
                    # Periodically announce objects
                    if frame_count % 30 == 0:
                        self.announce_objects()
                
                frame_count += 1
                
                # Calculate and print FPS every 100 frames
                if frame_count % 100 == 0:
                    end_time = time.time()
                    fps = 100 / (end_time - start_time)
                    print(f"FPS: {fps:.2f}")
                    start_time = time.time()
        
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1)
        self.camera.release()
        cv2.destroyAllWindows()
        print("Visual Assistant stopped.")

def main():
    try:
        # You can change the camera ID if you have multiple cameras
        # Default is 0 for the built-in webcam
        assistant = VisualAssistant(camera_id=0)
        assistant.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
