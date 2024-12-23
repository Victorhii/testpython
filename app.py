from flask import Flask, request, send_file
from ultralytics import YOLO
import os
import cv2
from io import BytesIO
from PIL import Image
import math

app = Flask(__name__)

# Set up a folder to save the uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your YOLO model
model_path = 'best.pt'
model = YOLO(model_path)

@app.route('/')
def index():
    return app.send_static_file('index.html')  # Serve the HTML file

@app.route('/process-image', methods=['POST'])
def process_image_route():
    # Get the uploaded image from the form
    image_file = request.files['image']
    
    if image_file:
        # Save the uploaded image to a temporary location
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)
        
        # Process the image using your YOLO model
        output_image = process_image(image_path)
        
        # Convert the processed image to a BytesIO object to send it in the response
        img_io = BytesIO()
        output_image.save(img_io, 'JPEG')
        img_io.seek(0)
        
        # Return the processed image
        return send_file(img_io, mimetype='image/jpeg')

def process_image(image_path):
    image = cv2.imread(image_path)

    # Process image with the YOLO model
    results = model(image, verbose=False)[0]
    
    threshold = 0.5  # Set threshold for object detection

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            # Define parameters for text and bounding boxes
            FONT_SCALE = 2e-3
            THICKNESS_SCALE = 1e-3
            height, width, _ = image.shape
            line_width = math.ceil(min(width, height) * THICKNESS_SCALE * 2)  # Double thickness
            font_size = min(width, height) * FONT_SCALE * 1.5  # Slightly larger font
            font_thickness = math.ceil(min(width, height) * THICKNESS_SCALE * 2)  # Double thickness

            # Draw a bold black rectangle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), line_width)

            # Add bold black text for the label
            label = results.names[int(class_id)]
            cv2.putText(image, label, (int(x1), int(y1 - 10)),  # Position above the box
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Convert the image with bounding boxes to a PIL Image
    output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(output_image)

    return output_image

if __name__ == '__main__':
    app.run(debug=True)
