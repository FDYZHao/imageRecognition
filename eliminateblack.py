#去除图像黑色背景
from PIL import Image 
import cv2
img2 = Image.open('tb.png')
#img1 = Image.open('./Amazing_RGB_2L.bmp')
# img1 = img1.convert('RGBA')
img2 = img2.convert('RGBA')
pixdata = img2.load()
for y in range(img2.size[1]):
    for x in range(img2.size[0]):
        if pixdata[x,y][0]==0 and pixdata[x,y][1]==0 and pixdata[x,y][2]==0:
           pixdata[x, y] = (255, 255, 255,0)

img2.show()
