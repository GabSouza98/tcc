import glob
import easyocr
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as PILImage
from wand.image import Image
import io
import os
# Imports the Google Cloud client library
from google.cloud import vision
from AVALIA_TEXTO import verifica_conformidade_pepsi

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r' '

#FUNCAO API GOOGLE VISION
def detect_text_only(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations         
        
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    
    if len(texts)>0:

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                        for vertex in texts[0].bounding_poly.vertices])        
        return vertices

    else:
        return "Impossível Identificar"




img_path = './V154/Pepsi/TRATADO/denoised_amostra6_5.png'
img = cv2.imread(img_path)
vertices = detect_text_only(img_path)

x1 = int(vertices[0][1:4])
y1 = int(vertices[0][5:8])
x2 = int(vertices[2][1:4])
y2 = int(vertices[2][5:8])

top_left = (x1,y1)
bottom_right = (x2,y2)

print(top_left)
print(bottom_right)

img = cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)

cv2.imshow("resultado", img)

cv2.waitKey(0)
