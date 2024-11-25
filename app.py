from functools import wraps
import sys
import os
from flask import Flask, render_template, redirect, request, url_for, session, send_from_directory
from ultralytics import YOLO
import uuid
# coming from pyrebase4
import pyrebase
import numpy as np
import threading
import cv2
# Firebase config
config = {
  "apiKey": "AIzaSyBUfO4Ofsdagt2nZvaN9csAp5Ik9AKtCiw",
  "authDomain": "infosect-a3e7a.firebaseapp.com",
  "projectId": "infosect-a3e7a",
  "storageBucket": "infosect-a3e7a.appspot.com",
  "messagingSenderId": "1038538489952",
  "appId": "1:1038538489952:web:142249a4ae77528b834305",
  "databaseURL": "https://infosect-a3e7a-default-rtdb.firebaseio.com"  # Add your databaseURL here
}

# Init Firebase
firebase = pyrebase.initialize_app(config)
# Auth instance
auth = firebase.auth()
# Realtime Database instance
db = firebase.database()

# New instance of Flask
app = Flask(__name__)
# Secret key for the session
app.secret_key = os.urandom(24)

# Load the YOLO model
model = YOLO("static/best.pt")  # Replace with the correct path to your model

# Ensure the uploads folder exists
UPLOAD_FOLDER = "static/uploads/"
PROCESSED_FOLDER = "static/processed/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
# Initialize Firebase Storage
storage = firebase.storage()


# Decorator to protect routes
def isAuthenticated(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for the variable that pyrebase creates
        if not auth.current_user:
            return redirect(url_for('signup'))
        return f(*args, **kwargs)
    return decorated_function
@app.route("/")
def index():
    # Get data from Firebase Realtime Database
    sensor_data = db.child("sensorData").get()

    # Check if the sensor_data is not None or empty
    if sensor_data.each():
        # Extract the values from the Firebase response
        sensor_info = sensor_data.val()

        # Get values for humidity, temperature, and rain
        humidity = sensor_info.get('humidity', 'N/A')  # Default to 'N/A' if not found
        temperature = sensor_info.get('temperature', 'N/A')  # Default to 'N/A' if not found
        rain = sensor_info.get('rain', 'N/A')  # Default to 'N/A' if not found

        # Render the HTML template with sensor data
        return render_template("index.html", humidity=humidity, temperature=temperature, rain=rain)

    # If no data is found in Firebase
    return "No sensor data available", 404

# Signup route
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        # Get the request form data
        email = request.form["email"]
        password = request.form["password"]
        try:
            # Create the user
            auth.create_user_with_email_and_password(email, password)
            # Login the user right away
            user = auth.sign_in_with_email_and_password(email, password)   
            # Session
            user_id = user['idToken']
            user_email = email
            session['usr'] = user_id
            session["email"] = user_email
            return redirect("/") 
        except:
            return render_template("login.html", message="The email is already taken, try another one, please")  

    return render_template("signup.html")

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Get the request data
        email = request.form["email"]
        password = request.form["password"]
        try:
            # Login the user
            user = auth.sign_in_with_email_and_password(email, password)
            # Set the session
            user_id = user['idToken']
            user_email = email
            session['usr'] = user_id
            session["email"] = user_email
            return redirect("/")  
        except:
            return render_template("login.html", message="Wrong Credentials")  

    return render_template("login.html")

# Logout route
@app.route("/logout")
def logout():
    # Remove the token setting the user to None
    auth.current_user = None
    # Also remove the session
    session.clear()
    return redirect("/")

from datetime import datetime  # Add this import at the top of your script

@app.route("/create", methods=["GET", "POST"])
@isAuthenticated
def create():
    if request.method == "POST":
        # Define the confidence threshold (e.g., 50%)
        confidence_threshold = 70.0

        # Get the uploaded file
        file = request.files["file"]
        if file:
            # Generate a unique filename for the processed file
            processed_filename = f"processed_{uuid.uuid4()}_{file.filename}"

            # Read the image using OpenCV directly from the uploaded file
            img_array = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Run YOLOv8 detection on the uploaded image
            results = model(img)
            detections = results[0].boxes.data  # Get detected objects

            # Initialize a dictionary to store counts per classification
            classification_counts = {}

            # Loop through the detections to draw bounding boxes and add labels
            for detection in detections:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = map(int, detection[:4])

                # Get the classification label and confidence score
                confidence = float(detection[4]) * 100  # Confidence score as percentage

                # Check if the confidence is above the threshold
                if confidence >= confidence_threshold:
                    # Get the classification label
                    class_id = int(detection[5])  # Index of the classification in YOLO
                    label = results[0].names[class_id]  # Get the class name using YOLO's label map

                    # Update the count for this classification
                    if label in classification_counts:
                        classification_counts[label] += 1
                    else:
                        classification_counts[label] = 1

                    # Draw the bounding box and label on the image
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
                    label_text = f"{label} {confidence:.2f}%"
                    cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save the processed image with bounding boxes
            processed_file_path = os.path.join(PROCESSED_FOLDER, processed_filename)
            cv2.imwrite(processed_file_path, img)

            # Get the current date and time
            detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS

            # Prepare the post data for Firebase
            post = {
                "file_name": processed_filename, 
                "classification_counts": classification_counts, 
                "author": session["email"],
                "detection_time": detection_time  # Add the detection time here
            }

            try:
                # Push the post data (including counts) to Firebase Realtime Database
                db.child("detected").push(post)
                return redirect(f"/show/{processed_filename}")
            except Exception as e:
                return render_template("create.html", message="Something went wrong: " + str(e))

    return render_template("create.html")

@app.route('/show/<filename>')
def show_image(filename):
    return render_template("show_image.html", filename=filename)

@app.route("/detected")
@isAuthenticated
def detected():
    allposts = db.child("detected").get()
    if allposts.val() is None:
        return render_template("list.html")
    else:
        # Convert the Firebase data into a format that can be used in the template
        detected_list = {item.key(): item.val() for item in allposts.each()}
        return render_template("list.html", detected_list=detected_list)

@app.route("/delete/<id>", methods=["POST"])
def delete(id):
    db.child("detected").child(id).remove()
    return redirect("/")

import time
import requests
import re

def sanitize_filename(url):
    base_filename = os.path.basename(url.split("?")[0])  # Extract filename from URL
    sanitized_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', base_filename)  # Sanitize filename
    return sanitized_filename

def detect_from_firebase():
    confidence_threshold = 70  # Set a low threshold for testing

    while True:
        uploaded_files_data = db.child("uploaded_files").get()
        
        if uploaded_files_data.val() is not None:
            sensor_info = uploaded_files_data.val()

            image_url = sensor_info.get('photo_url')

            if image_url:
                print(f"Processing image from URL: {image_url}")

                # Sanitize the URL to create a valid filename
                sanitized_filename = sanitize_filename(image_url)

                # Load the image from the URL
                resp = requests.get(image_url)
                img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"Error decoding image from URL: {image_url}")
                    continue

                # Run YOLOv8 detection on the image
                results = model(img)

                if not results:
                    print(f"No results for {image_url}.")
                    continue

                detections = results[0].boxes.data  # Get detected objects
                print(f"Detections for {image_url}: {detections}")  # Debugging output

                classification_counts = {}

                for detection in detections:
                    x1, y1, x2, y2 = map(int, detection[:4])
                    confidence = float(detection[4]) * 100  # Confidence score as percentage

                    if confidence >= confidence_threshold:
                        class_id = int(detection[5])  # Index of the classification in YOLO
                        label = results[0].names[class_id]  # Get the class name using YOLO's label map

                        if label in classification_counts:
                            classification_counts[label] += 1
                        else:
                            classification_counts[label] = 1

                        # Draw the bounding box and label on the image
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
                        label_text = f"{label} {confidence:.2f}%"
                        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save the processed image locally with bounding boxes
                processed_filename = f"processed_{uuid.uuid4()}_{sanitized_filename}"
                processed_file_path = f"static/processed/{processed_filename}"
                cv2.imwrite(processed_file_path, img)

                # Upload the processed image to Firebase Storage
                firebase_storage_path = f"processed_images/{processed_filename}"
                storage.child(firebase_storage_path).put(processed_file_path)

                # Get the public URL of the uploaded image
                processed_image_url = storage.child(firebase_storage_path).get_url(None)

                # Get the current detection time
                detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Prepare the post data to save to Firebase Realtime Database
                post = {
                    "file_name": processed_filename,
                    "classification_counts": classification_counts,
                    "author": "Automated System",  # Modify this as needed
                    "detection_time": detection_time,
                    "processed_image_url": processed_image_url  # Save the URL of the processed image
                }

                # Push the data to Firebase Realtime Database
                db.child("detected").push(post)
                print(f"Processed and pushed results for {image_url}")
                
                # Remove the processed file from the 'uploaded_files' node
                db.child("uploaded_files").remove()

        time.sleep(10)  # Wait before checking for new files

# Run detect_from_firebase in a background thread
def start_background_detection():
    detection_thread = threading.Thread(target=detect_from_firebase)
    detection_thread.daemon = True
    detection_thread.start()


# Run the Firebase detection loop
if __name__ == '__main__':
    start_background_detection()
    app.run(host='0.0.0.0', port=5000)

