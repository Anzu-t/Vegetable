from ultralytics import YOLO
from PIL import Image

import requests

url = 'https://www.hellozdrowie.pl/wp-content/uploads/2021/07/pomidory.jpg'
image = Image.open(requests.get(url, stream=True).raw)

model = YOLO("yolo11n.pt")
results = model.predict(source=image, conf=0.2, save=True)  # save plotted images