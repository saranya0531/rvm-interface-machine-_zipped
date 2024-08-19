from flask import Flask, render_template, Response,request,jsonify
import base64
from io import BytesIO
from PIL import Image
import qrhasher
import numpy as np
import cv2
from pymongo import MongoClient
import ssl

from roboflow import Roboflow
rf = Roboflow(api_key="DldOB8l12b56UbNgpwHZ")
project = rf.workspace().project("srp_r")
model = project.version(1).model

# MongoDB connection configuration
client = MongoClient("mongodb+srv://reversevendingmachinesrp:NtGqj7pusNnWf23F@rvm-details.xsk8gyn.mongodb.net/?retryWrites=true&w=majority")
db = client.get_database("db")
points_table_collection = db.get_collection("points_table")

app = Flask(__name__)

def gen_frames():  
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_image', methods=['POST'])
def process_image():
    image_data = request.form['image']
    image_data = image_data.replace('data:image/jpeg;base64,', '')
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    result = model.predict(image, confidence=40, overlap=30).json()
    print(result)

    # points_table = {}
    points = 0

    for i in result['predictions']:
        # points_table[i['class']]+=1
        match i['class']:
            case 'waterbottle':
                points+=10 #points for waterbottle
            case 'mask':
                points+=5 #points for mask
            case _:
                points+=0

    image = None
    if points == 0:
        image_path = 'static/nothing.gif'

    else:
        hash = qrhasher.qrhash(points)
        points_table_collection.insert_one({"hash":hash,"points":points})
        image_path = 'static/qr/newQr.png'

    # send the encoded image as a JSON response
    return jsonify(image_path=image_path)
    
    # for x in points_table:

if __name__ == '__main__':
    app.run(debug=True)
