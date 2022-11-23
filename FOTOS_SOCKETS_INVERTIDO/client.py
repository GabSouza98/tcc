import socket
import io
from PIL import Image, ImageFilter
import time
import glob
import easyocr
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as PILImage
from wand.image import Image
import io
import os
# Imports the Google Cloud client library
from google.cloud import vision
from AVALIA_TEXTO2 import verifica_conformidade_pepsi

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


# reader = easyocr.Reader(['en'], gpu=False) 

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('192.168.100.229', 12345))
print("CONECTADO")

BUFFER_SIZE = 4096

while True:
    
    print("AGUARDANDO NOVA FOTO")
    recv_data = client.recv(BUFFER_SIZE)                        # RECEBE O PRIMEIRO CHUNK DO CLIENT
    
    with open('imagem_recebida_nova.png', 'wb') as file:
        while recv_data:                                                   # ENQUANTO NÃO RECEBER VAZIO
            print("RECEBENDO DADOS")
            file.write(recv_data)                                   # ESCREVE O CHUNK RECEBIDO NO FILE_STREAM

            recv_data = client.recv(BUFFER_SIZE)                    # PEDE PARA RECEBER O PRÓXIMO CHUNK
            if recv_data == b"%IMAGE_COMPLETED%":                           # VERIFICAR SE CHEGOU O MARCADOR DE FIM  
                print("IMAGEM RECEBIDA")          
                break


    #################################################################### TRATAMENTOS
    print("COMEÇANDO O TRATAMENTO")

    start_time = time.time()
    img = cv2.imread('imagem_recebida_nova.png')
    img = sharpen(img)
    img = gray(img)    
    se  = cv2.getStructuringElement(cv2.MORPH_RECT , (40,50))
    bg  = cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
    img = cv2.divide(img, bg, scale=255)        
    img = cv2.fastNlMeansDenoising(img, None, 12,21,21)
    cv2.imwrite('imagem_recebida_tratada.png', img)      
            
    # print("INICIANDO A LEITURA COM EASYOCR")
    
    # result = reader.readtext('imagem_recebida_tratada.png',paragraph=True,
    #                                     x_ths=2,y_ths=1,                                    
    #                                     allowlist="VAL:PETPCR-1234567890LPS:")

    # img = cv2.imread('imagem_recebida_tratada.png')
    # kernel = np.ones((3,3),np.uint8)   
    # lista = []

    # for detection in result:
    #     x1, y1 = tuple([int(val) for val in detection[0][0]])
    #     x2, y2 = tuple([int(val) for val in detection[0][2]])        
    #     y1 = y1 - 12 
    #     y2 = y2 + 12
    #     x1 = x1 - 25
    #     x2 = x2 + 25
    #     top_left = (x1,y1)
    #     bottom_right = (x2,y2)
    #     text = detection[1]
    #     lista.append(text)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     img_crop = img.copy()
    #     img_crop = img_crop[(y1):(y2), (x1):(x2)]  #PEGA A PARTE QUE TEM A CODIFICAÇÃO
    #     img_resized = img_crop.copy()
    #     img_resized = cv2.resize(img_resized, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)         
    #     img_gray = gray(img_resized)                      
    #     img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 41)         
    #     img_denoised = cv2.fastNlMeansDenoising(img_thresh, None, 36,21,21)               
    #     img_dilated = cv2.dilate(img_denoised,kernel,iterations = 1)

    #     cv2.imwrite('imagem_recebida_cortada.png', img_dilated) #antes estava sem o TAB

    print("INICIANDO A LEITURA COM GOOGLE VISION API")
    kernel = np.ones((3,3),np.uint8) 
    img_path = 'imagem_recebida_tratada.png'
    img = cv2.imread(img_path)
    vertices = detect_text_only(img_path)    

    x1 = int(vertices[0][1:4].replace(',','').replace(')','').replace('(',''))
    y1 = int(vertices[0][5:8].replace(',','').replace(')','').replace('(',''))
    x2 = int(vertices[2][1:4].replace(',','').replace(')','').replace('(',''))
    y2 = int(vertices[2][5:8].replace(',','').replace(')','').replace('(',''))

    top_left = (x1,y1)
    bottom_right = (x2,y2)
    img_crop = img.copy()
    img_crop = img_crop[(y1):(y2), (x1):(x2)]  #PEGA A PARTE QUE TEM A CODIFICAÇÃO
    img_resized = img_crop.copy()
    img_resized = cv2.resize(img_resized, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)         
    img_gray = gray(img_resized)                      
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 91, 41)         
    img_denoised = cv2.fastNlMeansDenoising(img_thresh, None, 36,21,21)               
    img_dilated = cv2.dilate(img_denoised,kernel,iterations = 1)

    cv2.imwrite('imagem_recebida_cortada.png', img_dilated) #antes estava sem o TAB

    # print("COMEÇANDO O TRATAMENTO ARC DISTORTION")
            
    # with Image(filename='gray_recebido_cropped.png') as img:
    #     print("DISTORCENDO A IMAGEM")
    #     #img.resize(640, 480)        
    #     img.virtual_pixel = 'white'
    #     img.flip()
    #     args = (
    #         32,  # ArcAngle
    #         0,   # RotateAngle
    #     )
    #     img.distort('arc', args)
    #     img.flip()
    #     img_buffer=np.asarray(bytearray(img.make_blob()), dtype=np.uint8)
    #     retval = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)    
    #     #cv2.bitwise_not(retval, retval)
    #     #backtorgb = cv2.cvtColor(retval,cv2.COLOR_GRAY2RGB)
    #     cv2.imwrite('gray_recebido_cropped_distorcido.png', retval)  

    print("INICIANDO A LEITURA COM GOOGLE VISION")

    texto = detect_text('imagem_recebida_cortada.png')  
    conformidade = verifica_conformidade_pepsi(texto)
    tempo_total = "--- %s segundos ---" % round(time.time() - start_time, 3)
    texto_final = texto + '\n' + conformidade + '\n' + tempo_total   
    print(texto_final) 
    
    #################################################################### FIM TRATAMENTOS

    print("ENVIANDO RESULTADO AO SERVIDOR")
    msg = texto_final.encode('utf-8')
    client.send(msg)
    print("RESULTADO ENVIADO")
    os.remove('imagem_recebida_nova.png')
    os.remove('imagem_recebida_tratada.png')
    os.remove('imagem_recebida_cortada.png')
    time.sleep(2)


    

        

