import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import paddlehub as hub
import cv2
import numpy as np
from PIL import Image 

def characterRe():
    # 待预测图片
    test_img_path = ["D:\Desktop\ImageRecognition\img\jie1.PNG"]

    # 展示其中广告信息图片
    img1 = mpimg.imread(test_img_path[0]) 
    plt.figure(figsize=(10,10))
    plt.imshow(img1) 
    plt.axis('off') 
    plt.show()
    ocr = hub.Module(name="chinese_ocr_db_crnn_server") 
    np_images =[cv2.imread(image_path) for image_path in test_img_path] 

    results = ocr.recognize_text(
                        images=np_images,         # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
                        use_gpu=False,            # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
                        output_dir='ocr_result',  # 图片的保存路径，默认设为 ocr_result；
                        visualization=True,       # 是否将识别结果保存为图片文件；
                        box_thresh=0.5,           # 检测文本框置信度的阈值；
                        text_thresh=0.5)          # 识别中文文本置信度的阈值；

    for result in results:
        data = result['data']
        save_path = result['save_path']
        
        for infomation in data:
            print('text: ', infomation['text'], '\nconfidence: ', infomation['confidence'], '\ntext_box_position: ', infomation['text_box_position'])
            if('完工' in infomation['text']):
                print(infomation['text'])
                return infomation['text_box_position']

def calculateLength(x,y):
    a = abs(x[0]-y[0])
    b = abs(x[1]-y[1])
    l = (a**2+b**2)**0.5
    return l

# 根据标签和木板像素长宽、标签实际长宽计算木板实际长宽
# para
    #lalabelPxLocation:[[],[],[],[]] 标签四角像素坐标
    # labelActualLength：标签实际长宽  [l,w]
    # woodPxLength:木板像素长宽   [l,w]
def computeWoodLength(labelPxLocation,labelActualLength,woodPxLength):
    # print(labelPxLocation)
    l = calculateLength(labelPxLocation[0],labelPxLocation[1])
    w = calculateLength(labelPxLocation[1],labelPxLocation[2])
    actualL = (woodPxLength[0]*labelActualLength[0])/l
    actualW = (woodPxLength[1]*labelActualLength[1])/w
    return([actualL,actualW])

def canny(imgPath):      #边缘检测算法
    img = cv2.imread(imgPath,0)
    edges = cv2.Canny(img, 100, 200)
    # print(edges.size)
    # for i in range(0,533):
    #     for j in range(0,800):
    #         if(edges[i][j]!=0):
    #             print(edges[i][j],"++",i,"==",j)
    #             exit(0);
                
    plt.subplot(121)
    plt.imshow(img,cmap = 'gray')
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image')
    plt.xticks([]), plt.yticks([])

    plt.show()

def fit(data_x, data_y):   #最小二乘法模拟直线
    m = len(data_y)
    x_bar = np.mean(data_x)
    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0
    for i in range(m):
        x = data_x[i]
        y = data_y[i]
        sum_yx += y * (x - x_bar)
        sum_x2 += x ** 2
    # 根据公式计算w
    w = sum_yx / (sum_x2 - m * (x_bar ** 2))

    for i in range(m):
        x = data_x[i]
        y = data_y[i]
        sum_delta += (y - w * x)
    b = sum_delta / m
    return w, b

if __name__ == '__main__':
    x = np.arange(1, 17, 1)
    y = np.array([4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 10.86, 10.00, 10.20, 10.32, 10.42, 10.50, 11.55, 12.58, 13.60])
    w,b=fit(x,y)
    print(w,b)
    # path = "D:/Desktop/ImageRecognition11/img/pic2.jpg"
    # canny(path)
    # data = characterRe()
    # labelLength = [5,2]
    # woodPx = [800,800]
    # actual = computeWoodLength(data,labelLength,woodPx)
    # print(actual)
