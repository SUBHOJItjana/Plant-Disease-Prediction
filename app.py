#import necessary libraries
import http
import socketserver
from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import threading

# Load the trained model
model = load_model("D:/projects/plant/.ipynb_checkpoints/plant.h5")
print("@@ Model loaded")

def pred_cot_dieas(cott_plant):
    test_image = load_img(cott_plant, target_size=(150, 150))  # Load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255   # Convert image to np array
    test_image = np.expand_dims(test_image, axis=0)  # Change 3d to 4d
    
    result = model.predict(test_image).round(3)  # Predict disease or not
    print("@@ Raw result = ", result)
    
    pred = np.argmax(result)  # Get index of max value
    
    if pred == 0:
        return "Healthy Plant", 'healthy.html'
    elif pred == 1:
        return "Powdery Plant", 'powdery.html'
    else:
        return "Rust Plant", 'rust.html'

# Function to start HTTP server
def start_http_server(directory, port):
    os.chdir(directory)
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print("Serving at port", port)
        httpd.directory = directory
        httpd.serve_forever()

# Initialize Flask app
app = Flask(__name__)
app.debug = True  # Enable debug mode

# Set the directory containing your image files
directory = 'D:/projects/plant/static/user uploaded/'

# Start the HTTP server in a separate thread
server_thread = threading.Thread(target=start_http_server, args=(directory, 8001))  # Change the port to 8080
server_thread.start()

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files["image"]
        filename = file.filename
        print("@@ Input posted = ", filename)
        
        upload_dir = os.path.join("D:/projects/plant/static", "user_uploaded")
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        
        print("@@ Predicting class.......")
        # Assuming pred_cot_dieas is defined elsewhere
        pred, output_page = pred_cot_dieas(cott_plant=file_path)
        
        return render_template(output_page, pred_output=pred, user_image=file_path)

if __name__ == "__main__":
    app.run()
