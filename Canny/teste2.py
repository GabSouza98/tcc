import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageFilter

from PIL.ImageFilter import (
   BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE,
   EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
)

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread('FOTOS_GRF/4.jpg') #em BGR

im_rgb = Image.open('FOTOS_GRF/4.jpg')

im = im_rgb

# RESIZE
resized_im = im.resize((round(im_rgb.size[0]*0.5), round(im.size[1]*0.5)))
# resized_im.show()

threshold = 100
multiBands = resized_im.split()
redBand      = multiBands[0]
greenBand    = multiBands[1]
blueBand     = multiBands[2].point(lambda p: p > threshold and 255)
blueBand.show()

# gray = cv2.imread('teste4.png', cv2.IMREAD_GRAYSCALE)  #opcional, pois a imagem já esta em grayscale 

# edges = cv2.Canny(gray,50,150,apertureSize = 3)

# cv2.imwrite('edges-50-150.jpg',edges)
# minLineLength=300
# lines = cv2.HoughLinesP(image=edges,
#                         rho=1,
#                         theta=np.pi/180,
#                         threshold=100,
#                         lines=np.array([]),
#                         minLineLength=minLineLength,
#                         maxLineGap=80)


# filter_arr = []
# for i in lines:
#     for j in i:        
#         if abs(     abs(j[3] - j[1]) / abs(j[2]-j[0])   ) <=0.025:
#             filter_arr.append(True)
#         else:
#             filter_arr.append(False)
      
# horizontais = lines[filter_arr]
# print(horizontais)

# count = 0
# total = 0
# for i in horizontais:
#     for j in i:
#         total = total + j[3]
#         count = count + 1

# media_nivel = total/count

# print(media_nivel)

# backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

# a,b,c = horizontais.shape

# for i in range(a):

#     cv2.line(backtorgb, (horizontais[i][0][0], horizontais[i][0][1]), (horizontais[i][0][2], horizontais[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
#     cv2.imwrite('teste4_linhas.jpg',backtorgb)

# crop_img = backtorgb[550:round(media_nivel)-20, 400:1200]

# cv2.imwrite('crop4.jpg', crop_img)

# # im2arr = np.array(black_white) # im2arr.shape: height x width x channel
# # # arr2im = Image.fromarray(im2arr)
# # im_bgr = cv2.cvtColor(im2arr, cv2.COLOR_RGB2BGR)
# # im_bgr.show()


# # # converte para RGB
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # # Detecting Characters
# hImg, wImg, _ = crop_img.shape
# boxes = pytesseract.image_to_boxes(crop_img)

# for b in boxes.splitlines():
#     b = b.split(' ')
#     # print(b)
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     cv2.rectangle(crop_img, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)
#     cv2.putText(crop_img, b[0], (x, hImg - y + 25), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

# cv2.imshow('Result3', crop_img)
# cv2.waitKey(0)