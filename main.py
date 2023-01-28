import os
import cv2
# import filepath as filepath
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet

from glob import glob
from skimage import io
from shutil import copy
# from tensorflow._api.v1.keras import layers
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import TensorBoard
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.applications import InceptionResNetV2
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras import Model, Input
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Dropout, Flatten, Input
from keras.utils import load_img, img_to_array

import os
from os import listdir
import PIL
from PIL import Image
# import glob

path = glob('./images//*.xml')
labels_dict = dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])
for filename in path:

    info = xet.parse(filename)
    root = info.getroot()
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)

    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

df = pd.DataFrame(labels_dict)
df.to_csv('labels.csv',index=False)
df.head()

filename = df['filepath'][0]
def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('./images',filename_image)
    return filepath_image

getFilename(filename)

image_path = list(df['filepath'].apply(getFilename))
image_path[:10]

file_path = image_path[1] #path of our image N2.jpeg
img = cv2.imread(file_path) #read the image
# xmin-1804/ymin-1734/xmax-2493/ymax-1882
img = io.imread(file_path) #Read the image
fig = px.imshow(img)
fig.update_layout(width=600, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Figure 8 - N2.jpeg with bounding box')
fig.add_shape(type='rect',x0=1804, x1=2493, y0=1734, y1=1882, xref='x', yref='y',line_color='cyan')
#fig.show()

labels = df.iloc[:,1:].values
data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    h,w,d = img_arr.shape
    # Prepprocesing
    load_image = load_img(image,target_size=(224,224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr/255.0 # Normalization
    # Normalization to labels
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax = xmin/w,xmax/w
    nymin,nymax = ymin/h,ymax/h
    label_norm = (nxmin,nxmax,nymin,nymax) # Normalized output
    # Append
    data.append(norm_load_image_arr)
    output.append(label_norm)

# Convert data to array
X = np.array(data,dtype=np.float32)
y = np.array(output,dtype=np.float32)

# Split the data into training and testing set using sklearn.
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False, input_tensor=Input(shape=(224,224,3)))
# ---------------------
headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500,activation="relu")(headmodel)
headmodel = Dense(250,activation="relu")(headmodel)
headmodel = Dense(4,activation='sigmoid')(headmodel)

# ---------- model
model = Model(inputs=inception_resnet.input,outputs=headmodel)

# Compile model
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
model.summary()

tfb = TensorBoard('object_detection')
# history = model.fit(x=x_train,y=y_train,batch_size=10,epochs=180,validation_data=(x_test,y_test),callbacks=[tfb])

# model.save('./object_detection.h5')


# Load model
model = tf.keras.models.load_model('./object_detection.h5')
print('Model loaded Sucessfully')
def image():
    path = './TEST/C7.jpeg'
    # for images in os.listdir(path):
    image = load_img(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    image1 = load_img(path,target_size=(224,224))
    image_arr_224 = img_to_array(image1)/255.0  # Convert into array and get the normalized output

    # Size of the orginal image
    h,w,d = image.shape
    print('Height of the image =',h)
    print('Width of the image =',w)

    fig = px.imshow(image)
    fig.update_layout(width=700, height=500,  margin=dict(l=10, r=10, b=10, t=10), xaxis_title='Figure 13 - TEST Image')

    image_arr_224.shape

    test_arr = image_arr_224.reshape(1,224,224,3)
    test_arr.shape

    # Make predictions
    coords = model.predict(test_arr)
    coords

    # Denormalize the values
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    coords

    coords = coords.astype(np.int32)
    coords

    # Draw bounding on top the image
    xmin, xmax,ymin,ymax = coords[0]
    pt1 =(xmin,ymin)
    pt2 =(xmax,ymax)
    print(pt1, pt2)

    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    fig = px.imshow(image)
    fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10))
    # fig.show()

    # Create pipeline
    path = './TEST/C7.jpeg'

    # for images in os.listdir(path):
    def object_detection(path):
        # Read image
        image = load_img(path)  # PIL object
        image = np.array(image, dtype=np.uint8)  # 8 bit array (0,255)
        image1 = load_img(path, target_size=(224, 224))

        # Data preprocessing
        image_arr_224 = img_to_array(image1) / 255.0  # Convert to array & normalized
        h, w, d = image.shape
        test_arr = image_arr_224.reshape(1, 224, 224, 3)

        # Make predictions
        coords = model.predict(test_arr)

        # Denormalize the values
        denorm = np.array([w, w, h, h])
        coords = coords * denorm
        coords = coords.astype(np.int32)

        # Draw bounding on top the image
        xmin, xmax, ymin, ymax = coords[0]
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        print(pt1, pt2)
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
        return image, coords


    image, cods = object_detection(path)

    fig = px.imshow(image)
    fig.update_layout(width=700, height=500, margin=dict(l=10, r=10, b=10, t=10), xaxis_title='Figure 14')
    # fig.show()

    img = np.array(load_img(path))
    xmin ,xmax,ymin,ymax = cods[0]
    roi = img[ymin:ymax,xmin:xmax]
    fig = px.imshow(roi)
    fig.update_layout(width=350, height=250, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='xx')
    # fig.show()

    # extract text from image
    text = pt.image_to_string(roi)
    # print(text)

image()

# Path='TEST'
# for filename in glob.glob(Path):
#     path=Image.open(filename)
#     image()

# parsing
def parsing(path):
    parser = xet.parse(path).getroot()
    name = parser.find('filename').text
    filename = f'./images/{name}'

    # width and height
    parser_size = parser.find('size')
    width = int(parser_size.find('width').text)
    height = int(parser_size.find('height').text)

    return filename, width, height


df[['filename', 'width', 'height']] = df['filepath'].apply(parsing).apply(pd.Series)
df.head()

# center_x, center_y, width , height
df['center_x'] = (df['xmax'] + df['xmin'])/(2*df['width'])
df['center_y'] = (df['ymax'] + df['ymin'])/(2*df['height'])

df['bb_width'] = (df['xmax'] - df['xmin'])/df['width']
df['bb_height'] = (df['ymax'] - df['ymin'])/df['height']
df.head()

### split the data into train and test
df_train = df.iloc[:200]
df_test = df.iloc[200:]

train_folder = './yolov5/data_images/train'

values = df_train[['filename', 'center_x', 'center_y', 'bb_width', 'bb_height']].values
for fname, x, y, w, h in values:
    image_name = os.path.split(fname)[-1]
    txt_name = os.path.splitext(image_name)[0]

    dst_image_path = os.path.join(train_folder, image_name)
    dst_label_file = os.path.join(train_folder, txt_name + '.txt')

    # copy each image into the folder
    copy(fname, dst_image_path)

    # generate .txt which has label info
    label_txt = f'0 {x} {y} {w} {h}'
    with open(dst_label_file, mode='w') as f:
        f.write(label_txt)

        f.close()
test_folder = './yolov5/data_images/test'

values = df_test[['filename', 'center_x', 'center_y', 'bb_width', 'bb_height']].values
for fname, x, y, w, h in values:
    image_name = os.path.split(fname)[-1]
    txt_name = os.path.splitext(image_name)[0]

    dst_image_path = os.path.join(test_folder, image_name)
    dst_label_file = os.path.join(test_folder, txt_name + '.txt')

    # copy each image into the folder
    copy(fname, dst_image_path)

    # generate .txt which has label info
    label_txt = f'0 {x} {y} {w} {h}'
    with open(dst_label_file, mode='w') as f:
        f.write(label_txt)

        f.close()

# import torch
# from GPUtil import showUtilization as gpu_usage
# from numba import cuda
# def free_gpu_cache():
#     print("Initial GPU Usage")
#     gpu_usage()
#
#     torch.cuda.empty_cache()
#
#     cuda.select_device(0)
#     cuda.close()
#     cuda.select_device(0)
#
#     print("GPU Usage after emptying the cache")
#     gpu_usage()
#
# free_gpu_cache()


# settings
INPUT_WIDTH =  640
INPUT_HEIGHT = 640

# LOAD THE IMAGE
img = io.imread('./TEST/N96.jpeg')

fig = px.imshow(img)
fig.update_layout(width=700, height=400, margin=dict(l=10, r=10, b=10, t=10))
fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
# fig.show()

# LOAD YOLO MODEL
net = cv2.dnn.readNetFromONNX('./yolov5/runs/train/Model2/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def get_detections(img, net):
    # 1.CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections


def non_maximum_supression(input_image, detections):
    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE

    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]  # confidence of detecting license plate
        if confidence > 0.4:
            class_score = row[5]  # probability score of license plate
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    # 4.2 NMS
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)

    return boxes_np, confidences_np, index

def drawings(image,boxes_np,confidences_np,index):
    # 5. Drawings
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        license_text = extract_text(image,boxes_np[ind])


        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+25),(0,0,0),-1)


        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
        cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)

    return image

# predictions flow with return result
def yolo_predictions(img,net):
    # step-1: detections
    input_image, detections = get_detections(img,net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    # step-3: Drawings
    result_img = drawings(img,boxes_np,confidences_np,index)
    return result_img


# extrating text
def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y + h, x:x + w]

    if 0 in roi.shape:
        return 'no number'

    else:
        text = pt.image_to_string(roi)
        text = text.strip()
        print(text)

        return text

# test
# img = io.imread('./TEST/N96.jpeg')
# results = yolo_predictions(img,net)
#
# fig = px.imshow(img)
# fig.update_layout(width=700, height=400, margin=dict(l=10, r=10, b=10, t=10))
# fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
# fig.show()


cap = cv2.VideoCapture('./TEST/V2.mp4')
while True:
    ret, frame = cap.read()

    if ret == False:
        print('Unable to read video')
        break

    results = yolo_predictions(frame,net)

    cv2.namedWindow('YOLO',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('YOLO',results)
    if cv2.waitKey(30) == 27 :
        break

cv2.destroyAllWindows()
cap.release()




