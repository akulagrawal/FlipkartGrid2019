import cv2
import numpy as np
import pandas as pd

df = pd.read_csv('./data/OriginalData/train_labels.csv')
n = len(df)

mask = np.random.randint(0, 13999, 4000)
# print(len(mask))
test_df = df.ix[mask]
train_df = df.drop(df.index[mask])
train_df.to_csv("./data/train_labels.csv")
test_df.to_csv("./data/test_labels.csv")

# for i in mask:
# 	img_name = df.loc[i, 'filename']
# 	img = cv2.imread("./image/"+str(img_name))
# 	height[i], width[i] = img.shape[:2]

# df.insert(1, 'width', width)
# df.insert(2, 'height', height)
# df.insert(3, 'class', label)
# df.to_csv('new_train_labels.csv')


# import cv2
# import numpy as np
# import pandas as pd

# df = pd.read_csv('./data/train_labels.csv')
# n = len(df)
# height = np.zeros(n)
# width = np.zeros(n)

# label = ['object']*n
# print(n)
# for i in range(n):
# 	print(str(i)+"/"+str(n))
# 	img_name = df.loc[i, 'filename']
# 	img = cv2.imread("./image/"+str(img_name))
# 	height[i], width[i] = img.shape[:2]

# df.insert(1, 'width', width)
# df.insert(2, 'height', height)
# df.insert(3, 'class', label)
# df.to_csv('new_train_labels.csv')
