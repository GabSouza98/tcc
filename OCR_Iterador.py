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
    
    return texts[0].description


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

print("COMEÇANDO O TRATAMENTO")
count = 1
for IMAGE_PATH in glob.glob('./BRIGHT70_BATELADA1/*brightness70*.png'):

    print("TRATANDO A IMAGEM ", count)
    img = cv2.imread(IMAGE_PATH)
    img = sharpen(img)
    img = gray(img)
    cv2.imwrite('./BRIGHT70_BATELADA1/TRATADO/gray_{}.png'.format(count), img)
    img = black_and_white(img, 110)
    cv2.imwrite('./BRIGHT70_BATELADA1/TRATADO/binario_{}.png'.format(count), img)
    count += 1
    

print("COMEÇANDO O TRATAMENTO BARREL DISTORTION")
count = 1
for IMAGE_PATH in glob.glob('./BRIGHT70_BATELADA1/TRATADO/*binario*.png'):

    with Image(filename=IMAGE_PATH) as img:
        print("DISTORCENDO A IMAGEM ", count)
        #img.resize(640, 480)        
        img.virtual_pixel = 'white'
        args = (
            0.1,  # A melhores parametros que eu achei...
            0.1,  # B
            0.1,  # C
            0.6,  # D
        )
        img.distort('barrel', args)
        img_buffer=np.asarray(bytearray(img.make_blob()), dtype=np.uint8)
        retval = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)    
        #cv2.bitwise_not(retval, retval)
        backtorgb = cv2.cvtColor(retval,cv2.COLOR_GRAY2RGB)
        cv2.imwrite('./BRIGHT70_BATELADA1/TRATADO/distorcido_{}.png'.format(count), backtorgb)
        count += 1       


print("INICIANDO A LEITURA COM EASYOCR")

reader = easyocr.Reader(['en'], gpu=False)
count=1
lista = []
for IMAGE_PATH in glob.glob('./BRIGHT70_BATELADA1/TRATADO/*gray*.png'):

    print("LENDO COM EASYOCR A IMAGEM ", count)
    result = reader.readtext(IMAGE_PATH)
    img = cv2.imread(IMAGE_PATH)

    for detection in result:
        top_left = tuple([int(val) for val in detection[0][0]])
        bottom_right = tuple([int(val) for val in detection[0][2]])
        text = detection[1]
        lista.append(text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.rectangle(img, top_left, bottom_right, (0,255,0), 3)
        img = cv2.putText(img, text, top_left, font, 1, (0,0,255), 2, cv2.LINE_AA)

    cv2.imwrite('./BRIGHT70_BATELADA1/TRATADO/EASYOCR/gray_{}_ocr.png'.format(count), img)
    
    with open('./BRIGHT70_BATELADA1/TRATADO/EASYOCR/resultados_easyocr.txt', 'a+') as f:
        f.write('Resultado {}\n'.format(count))
        for line in lista:
            f.write(line)
            f.write('\n')
        f.write('###########\n')
    lista.clear()
    count += 1

print("INICIANDO A LEITURA COM GOOGLE VISION")
count = 1
for IMAGE_PATH in glob.glob('./BRIGHT70_BATELADA1/TRATADO/*gray*.png'):
    
    print("LENDO COM GOOGLE VISION A IMAGEM ", count)
    texto = detect_text(IMAGE_PATH)
    with open('./BRIGHT70_BATELADA1/TRATADO/GOOGLEVISION/resultados_googlevision.txt', 'a+') as f:
        f.write('Resultado {}\n'.format(count))
        f.write(texto)
        f.write('\n')
        f.write('###########\n')    
    count += 1

print("PROGRAMA EXECUTADO COM SUCESSO")

    

        
            

    
    

    
