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
# from AVALIA_TEXTO import verifica_conformidade_pepsi

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r' '

#FUNCAO API GOOGLE VISION
def detect_text(path):
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
        return texts[0].description
    else:
        return "Impossível Identificar"


#FUNÇÃO DE ROTAÇÃO
def rotate(image, angle, center = None, scale = 1.0):   
   (h, w) = image.shape[:2]

   if center is None:
       center = (w / 2, h / 2)

   # Perform the rotation
   M = cv2.getRotationMatrix2D(center, angle, scale)
   rotated = cv2.warpAffine(image, M, (w, h))
   return rotated

#FUNÇÃO DE SHARPEN
def sharpen(image):
   kernel = np.array([[-1,-1,-1], 
                      [-1, 9,-1],
                      [-1,-1,-1]])
   sharpened = cv2.filter2D(image, -1, kernel) # applying the sharpening kernel to the input image   
   return sharpened

def gray(image):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
   return gray

def blur(image, n = 5):   
   blurred = cv2.GaussianBlur(image, (n, n), 0)
   return blurred

def black_and_white(image, threshold):
   (T, thresh_binary) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
   return thresh_binary

def retorna_nome_amostra(caminho):
    caminho = caminho.split('.')  #SEPARA NO PONTO        ['', '/V15/Fruki\\amostra1_1', 'png']
    caminho = caminho[1]          #PEGA A PARTE DO MEIO   /V15/Fruki\amostra2_4
    caminho = caminho[-10:]       #PEGA DE AMOSTRA +      amostra1_1 
    return caminho
  
  
#Create MSER object
mser = cv2.MSER_create()

#Your image path i-e receipt path
img = cv2.imread('./V154/Pepsi/TRATADO/denoised_amostra6_5.png')
vis = img.copy()

#detect regions in image
regions, _ = mser.detectRegions(img)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(vis, hulls, 1, (0, 255, 0))

cv2.imshow('img', vis)

cv2.waitKey(0)

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for contour in hulls:

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("text only", text_only)

cv2.waitKey(0)

print("PROGRAMA EXECUTADO COM SUCESSO")