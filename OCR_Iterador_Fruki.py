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
from AVALIA_TEXTO2 import verifica_conformidade_fruki


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'testandoocr-be7a4b60a756.json'

#FUNCAO API GOOGLE VISION
def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image, image_context={"language_hints": ["pt"]})
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


# Calculate skew angle of an image
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()    
    blur = cv2.GaussianBlur(newImage, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 20))
    dilate = cv2.dilate(thresh, kernel, iterations=3)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

    # Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)


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


# print("COMEÇANDO O TRATAMENTO")
# for IMAGE_PATH in glob.glob('./V154/Fruki/*.png'):

#     amostra = retorna_nome_amostra(IMAGE_PATH)
#     print("TRATANDO A IMAGEM ", amostra)
#     img = cv2.imread(IMAGE_PATH)
#     img = sharpen(img)    
#     img = gray(img)
#     se  = cv2.getStructuringElement(cv2.MORPH_RECT , (40,50))
#     bg  = cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
#     img = cv2.divide(img, bg, scale=255)        
#     img = cv2.fastNlMeansDenoising(img, None, 12,21,21)
#     cv2.imwrite(f'./V154/Fruki/TRATADO/denoised_{amostra}.png', img)   #PADRÃO PARA TESTAR NO GOOGLE API    
  

# print("INICIANDO A LEITURA INICIAL COM EASYOCR")
# reader = easyocr.Reader(['en'], gpu=False)
# kernel = np.ones((3,3),np.uint8)   
# lista = []
# for IMAGE_PATH in glob.glob('./V154/Fruki/TRATADO/*denoised*.png'):

#     amostra = retorna_nome_amostra(IMAGE_PATH)
#     print("LENDO COM EASYOCR A IMAGEM ", amostra)
    
#     result = reader.readtext(IMAGE_PATH,paragraph=True,
#                                     x_ths=2,y_ths=1,                                    
#                                     allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890:")
                                    
#     img = cv2.imread(IMAGE_PATH)

#     for detection in result:
#         x1, y1 = tuple([int(val) for val in detection[0][0]])
#         x2, y2 = tuple([int(val) for val in detection[0][2]])        
#         y1 = y1 - 2 
#         y2 = y2 + 2
#         x1 = x1 - 6
#         x2 = x2 + 10
#         top_left = (x1,y1)
#         bottom_right = (x2,y2)
#         text = detection[1]
#         lista.append(text)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         img_crop = img.copy()
#         img_crop = img_crop[(y1):(y2), (x1):(x2)]  #PEGA A PARTE QUE TEM A CODIFICAÇÃO
#         img_resized = img_crop.copy()
#         img_resized = cv2.resize(img_resized, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)         
#         img_gray = gray(img_resized)                      
#         img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 41)         
#         img_denoised = cv2.fastNlMeansDenoising(img_thresh, None, 36,21,21)               
#         img_dilated = cv2.dilate(img_denoised,kernel,iterations = 1)                  

#         # img = cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)
#         # img = cv2.putText(img, text, top_left, font, 1, (0,0,255), 2, cv2.LINE_AA)
    
#     cv2.imwrite(f'./V154/Fruki/TRATADO/dilated_{amostra}.png', img_dilated)   
    
#     with open('./V154/Fruki/TRATADO/EASYOCR/resultados_easyocr.txt', 'a+') as f:
#         f.write(f'Resultado dilated {amostra}\n')
#         for line in lista:
#             f.write(line)
#             f.write('\n')
#         f.write('###########\n\n')
#     lista.clear()


# distorcer = ['dilated']
# print("COMEÇANDO O TRATAMENTO ARC DISTORTION")
# for tratamento in distorcer:
#     for IMAGE_PATH in glob.glob(f'./V154/Fruki/TRATADO/*{tratamento}*.png'):

#         amostra = retorna_nome_amostra(IMAGE_PATH)
#         with Image(filename=IMAGE_PATH) as img:
#             print("DISTORCENDO A IMAGEM ", tratamento, " ", amostra)
#             #img.resize(640, 480)        
#             img.virtual_pixel = 'white'
#             img.flip()
#             args = (
#                 30,  # ArcAngle
#                 0,   # RotateAngle
#             )
#             img.distort('arc', args)
#             img.flip()
#             img_buffer=np.asarray(bytearray(img.make_blob()), dtype=np.uint8)
#             retval = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)    
#             #cv2.bitwise_not(retval, retval)
#             #backtorgb = cv2.cvtColor(retval,cv2.COLOR_GRAY2RGB)         

#             img_deskewed = deskew(retval)     
#             cv2.imwrite(f'./V154/Fruki/TRATADO/distorted_{amostra}.png', retval)  
#             cv2.imwrite(f'./V154/Fruki/TRATADO/deskewed_{amostra}.png', img_deskewed)  



print("INICIANDO A LEITURA COM GOOGLE VISION")
for IMAGE_PATH in glob.glob('./V154/Fruki/TRATADO/*denoised*.png'):
    
    amostra = retorna_nome_amostra(IMAGE_PATH)
    print("LENDO COM GOOGLE VISION A IMAGEM ", amostra)
    texto = detect_text(IMAGE_PATH)
    with open('./V154/Fruki/TRATADO/GOOGLEVISION/resultados_googlevision.txt', 'a+', encoding="utf-8") as f:
        f.write(f'Resultado Final denoised GoogleVision {amostra}\n')
        f.write(texto)
        f.write("\n")
        f.write(verifica_conformidade_fruki(texto))                
        f.write('\n###########\n\n')    


# print("INICIANDO A LEITURA FINAL COM EASYOCR")
# reader = easyocr.Reader(['en'], gpu=False)
# kernel = np.ones((3,3),np.uint8)   
# lista = []
# for IMAGE_PATH in glob.glob('./V154/Pepsi/TRATADO/*dilated*.png'):

#     amostra = retorna_nome_amostra(IMAGE_PATH)
#     print("LENDO COM EASYOCR A IMAGEM ", amostra)
#     result = reader.readtext(IMAGE_PATH,paragraph=True,
#                                     x_ths=2,y_ths=1,                                    
#                                     allowlist="VAL:PETPCR-1234567890LPS:")
#     img = cv2.imread(IMAGE_PATH)

#     for detection in result:
#         x1, y1 = tuple([int(val) for val in detection[0][0]])
#         x2, y2 = tuple([int(val) for val in detection[0][2]])        
#         y1 = y1 - 12 
#         y2 = y2 + 12
#         x1 = x1 - 25
#         x2 = x2 + 25
#         top_left = (x1,y1)
#         bottom_right = (x2,y2)
#         text = detection[1]
#         lista.append(text)
#         font = cv2.FONT_HERSHEY_SIMPLEX               
        
#         img = cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)
#         #img = cv2.putText(img, text, top_left, font, 1, (0,0,255), 2, cv2.LINE_AA)

#     cv2.imwrite(f'./V154/Pepsi/TRATADO/EASYOCR/FINAL/ocr_{amostra}.png', img)    
      
    
#     with open('./V154/Pepsi/TRATADO/EASYOCR/FINAL/resultados_easyocr.txt', 'a+') as f:
#         f.write(f'Resultado dilated EasyOCR {amostra}\n')
#         for line in lista:
#             f.write(line)
#             f.write('\n')
#         f.write('###########\n\n')
#     lista.clear()



    

print("PROGRAMA EXECUTADO COM SUCESSO")