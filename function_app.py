import logging
import torch
import cv2
import numpy as np
import azure.functions as func
import json
from urllib.request import urlopen

app = func.FunctionApp()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5 #NMS Confidence Threshold
model.multi_label = False
model.classes = [0] # Detect people only
model.max_det = 1000

@app.function_name(name="DetectFromFile")
@app.route(route="DetectFromFile", methods=["POST"])
def detect_from_file(req: func.HttpRequest):
    output = []

    for input_file in req.files.values():
        filename = input_file.filename        
        data = {}        
        data["fileName"] = filename
        data["results"] = []

        logging.info('Processing File: %s' % filename)

        image_bytes = np.asarray(bytearray(input_file.stream.read()), dtype="uint8")
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        with torch.no_grad():
            results = model(image)
            df = results.pandas().xyxy[0]            
            df = df.loc[df['class']==0]
            
            for tup in df[['name', 'confidence']].itertuples():                
                obj = {}
                obj['detection_id'] = tup[0]
                obj["class"] = tup.name
                obj["confidence"] = tup.confidence
                data['results'].append(obj)

        output.append(data)            
    
    return func.HttpResponse(json.dumps(output), mimetype="application/json")

@app.function_name(name="DetectFromUrl")
@app.route(route="DetectFromUrl", methods=["GET"])
def detect_from_url(req: func.HttpRequest):
    output = []
    image_url = req.params.get('img')
    print(image_url)
    if not image_url:
        return func.HttpResponse("Url must contain an 'img' query parameter with a valid url", status_code=400)
    
    data = {}        
    data["fileName"] = image_url
    data["results"] = []

    with urlopen(image_url) as input_img:
        image_bytes = np.asarray(bytearray(input_img.read()), dtype="uint8")
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        with torch.no_grad():
            results = model(image)
            df = results.pandas().xyxy[0]            
            df = df.loc[df['class']==0]
            
            for tup in df[['name', 'confidence']].itertuples():                
                obj = {}
                obj['detection_id'] = tup[0]
                obj["class"] = tup.name
                obj["confidence"] = tup.confidence
                data['results'].append(obj)

        output.append(data)
    print(output)
    return func.HttpResponse(json.dumps(output), mimetype="application/json")                
    