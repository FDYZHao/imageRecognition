import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import paddlehub as hub
import cv2

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

if __name__ == '__main__':
    data = characterRe()
    labelLength = [5,2]
    woodPx = [800,800]
    # actual = computeWoodLength(data,labelLength,woodPx)
    # print(actual)