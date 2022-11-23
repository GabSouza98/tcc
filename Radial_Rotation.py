import cv2
import numpy as np
import pytesseract
from PIL import Image as PILImage
from wand.image import Image

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

#RECEBENDO A IMAGEM TRATADA
#bright70 = cv2.imread('Brightness_70_tratado.png')

# USANDO O DISTORT BARREL
# with Image(filename='./V15/Fruki/TRATADO/gray_cropped_amostra2_1.png') as img:
#     #img.resize(640, 480)
#     #img.background_color = Color('skyblue')
#     img.virtual_pixel = 'white'
#     args = (
#         0.2,  # A melhores parametros que eu achei...
#         0.2,  # B
#         0.2,  # C
#         0.5,  # D
#     )
#     img.distort('barrel', args)
#     img_buffer=np.asarray(bytearray(img.make_blob()), dtype=np.uint8)
#     retval = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)    

#     #cv2.bitwise_not(retval, retval)

#     backtorgb = cv2.cvtColor(retval,cv2.COLOR_GRAY2RGB)

    # hImg, wImg, _ = backtorgb.shape

    # boxes = pytesseract.image_to_boxes(backtorgb)
    # for b in boxes.splitlines():
    #     b = b.split(' ')
    #     print(b)
    #     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    #     cv2.rectangle(backtorgb, (x, hImg - y), (w, hImg - h), (0, 255, 0), 1)
    #     cv2.putText(backtorgb, b[0], (x, hImg - y + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
 
   
# with Image(filename='Brightness_70_tratado.png') as img:    
#     #img.background_color = Color('skyblue')
#     img.virtual_pixel = 'white'
#     lens = 40
#     film = 160
#     args = (
#         lens/film * 180/math.pi,
#         200,
#         302,
#         50,
#         330,
#         400,
#     )
#     img.distort('cylinder_2_plane', args)
#     matrix = np.array(img)

#cv2.imwrite("./V15/Fruki/TRATADO/gray_amostra2_1_barrel.png", backtorgb)

# with Image(filename='./V15/Fruki/TRATADO/gray_cropped_amostra2_1.png') as img:
#     #img.resize(640, 480)
#     #img.background_color = Color('skyblue')
#     img.virtual_pixel = 'white'
#     args = (
#         0.2,  # A melhores parametros que eu achei...
#         0.0,  # B
#         0.0,  # C
#         1.2,  # D
#     )
#     img.distort('barrel', args)
#     img_buffer=np.asarray(bytearray(img.make_blob()), dtype=np.uint8)
#     retval = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)    

#     #cv2.bitwise_not(retval, retval)

#     backtorgb = cv2.cvtColor(retval,cv2.COLOR_GRAY2RGB)


with Image(filename='./V15/Fruki/TRATADO/gray_cropped_amostra5_2.png') as img:
    #img.resize(640, 480)
    #img.background_color = Color('skyblue')
    img.flip()
    img.virtual_pixel = 'white'
    args = (
        32,  # ArcAngle
        0,   # RotateAngle
    )
    img.distort('arc', args)
    img.flip()
    img_buffer=np.asarray(bytearray(img.make_blob()), dtype=np.uint8)
    retval = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)  
    #cv2.bitwise_not(retval, retval)
    backtorgb2 = cv2.cvtColor(retval,cv2.COLOR_GRAY2RGB)






#cv2.imshow("Barrel", backtorgb)

# def sharpen(image):
#    kernel = np.array([[-1,-1,-1], 
#                       [-1, 9,-1],
#                       [-1,-1,-1]])
#    sharpened = cv2.filter2D(image, -1, kernel) # applying the sharpening kernel to the input image   
#    return sharpened

# def black_and_white(image, threshold):
#    (T, thresh_binary) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
#    return thresh_binary

# backtorgb2 = cv2.imread('./V15/Fruki/TRATADO/gray_resized_amostra5_2.png')
# backtorgb2 = sharpen(backtorgb2)
# backtorgb2 = black_and_white(backtorgb2, 80)

cv2.imshow("Arc", backtorgb2)



# SALVAR IMAGEM
# with open('Brightness_50_tratado.png', 'rb') as File:
#     BinaryData = File.read()

# print(type(BinaryData))
# print(BinaryData)

# with open('SALVADO_IMAGEM.png', 'wb') as File:
#     File.write(BinaryData)
#     File.close()

cv2.waitKey(0)
cv2.destroyAllWindows()