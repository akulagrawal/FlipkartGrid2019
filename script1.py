
##################Preprocess csv file######################
import cv2
import numpy as np
import pandas as pd

df = pd.read_csv('./data/training.csv')
df.rename(columns={'image_name': 'filename', 'x1': 'xmin', 'x2': 'xmax', 'y1': 'ymin', 'y2': 'ymax'}, inplace=True)
n = len(df)

height = np.zeros(len(df))
width = np.zeros(len(df))

label = ['object']*n
for i in range(n):
	print(str(i)+'/'+str(n))
	img_name = df.loc[i, 'filename']
	img = cv2.imread("./images/"+str(img_name))
	height[i], width[i] = int(img.shape[0]), int(img.shape[1])
df.insert(1, 'width', width)
df.insert(2, 'height', height)
df.insert(3, 'class', label)
df['width']=df['width'].astype(int)
df['height']=df['height'].astype(int)
print(df.head())
df.to_csv('./data/train_labels.csv', index=False)
