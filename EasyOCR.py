import easyocr
import cv2
from easyocr.recognition import ListDataset
import numpy as np
from matplotlib import pyplot as plt

#IMAGE_PATH = './V15/Fruki/TRATADO/gray_resized_amostra5_2.png'
IMAGE_PATH = 'denoised.png'
#IMAGE_PATH = './V15/Pepsi/amostra1_6.png'
#IMAGE_PATH = 'Imagem_Distorcida_Radialmente.png'
#IMAGE_PATH = 'Brightness_70_tratado.png'
#IMAGE_PATH = 'Imagem_Distorcida_Radialmente_Recortada.png'


reader = easyocr.Reader(['en'], gpu=False, recognizer=True)
# result = reader.detect(IMAGE_PATH)                                               # ONLY DETECT
result = reader.readtext(IMAGE_PATH,paragraph=True,                            # READ TEXT
                                    x_ths=2,y_ths=1,                                    
                                    allowlist="VAL:PETPCR-1234567890LPS:")


#result = [numero da deteccao] [ [[pardecoordenadas][par de coord][coord][coord]] , [texto detectado] , [acuracia] ] 

img = cv2.imread(IMAGE_PATH)

# READTEXT METHOD
for detection in result:
    top_left = tuple([int(val) for val in detection[0][0]])
    bottom_right = tuple([int(val) for val in detection[0][2]])
    x1, y1 = top_left
    x2, y2 = bottom_right
    y1 = y1 - 8 
    y2 = y2 + 10
    x1 = x1 - 20
    x2 = x2 + 20
    top_left = (x1,y1)
    bottom_right = (x2,y2)
    img_crop = img.copy()
    img_crop = img_crop[(y1):(y2), (x1):(x2)]
    text = detection[1]
    print(text)
    font = cv2.FONT_HERSHEY_SIMPLEX   
    img = cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)
    img = cv2.putText(img, text, top_left, font, 1, (0,0,255), 2, cv2.LINE_AA)    
cv2.imshow("OCR", img)
#cv2.imwrite('denoised_cropped.png', img_crop)

# FOR DETECT METHOD
# for i in result[0]:    
#     x_min = i[0]
#     x_max = i[1]
#     y_min = i[2]
#     y_max = i[3]

#     top_left = (x_min,y_min)
#     bottom_right = (x_max,y_max) 
#     horizontal_boxes = img.copy()
#     horizontal_boxes = cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)


# free_boxes = img.copy()
# for i in result[1]:
#     array = np.array([i[0],i[1],i[2],i[3]],np.int32)
#     array = array.reshape((-1, 1, 2))
    
#     print(array)   

#     free_boxes = cv2.polylines(free_boxes, 
#                 [array], 
#                 isClosed = True,
#                 color = (0,255,0),
#                 thickness = 2,
#                 lineType=cv2.LINE_AA)
   
    
# cv2.imshow("Horizontal List", horizontal_boxes)
# cv2.imshow("Free-boxes", free_boxes)

cv2.waitKey(0)
cv2.destroyAllWindows()
