# import cv2
# import numpy as np
# import pandas as pd

# df = pd.read_csv('./large_data/OriginalData/train_labels.csv')
# n = len(df)

########### Create Train and Test csv files#############
# mask = np.random.randint(0, 199, 40)
# test_df = df.ix[mask]
# train_df = df.drop(df.index[mask])
# train_df.to_csv("./data/train_labels.csv", index=False)
# test_df.to_csv("./data/test_labels.csv", index=False)

############# Add width, height and class to csv########
# for i in mask:
# 	img_name = df.loc[i, 'filename']
# 	img = cv2.imread("./image/"+str(img_name))
# 	height[i], width[i] = img.shape[:2]

# df.insert(1, 'width', width)
# df.insert(2, 'height', height)
# df.insert(3, 'class', label)
# df.to_csv('new_train_labels.csv', index=False)


import cv2
import numpy as np
import pandas as pd

df = pd.read_csv('./data/labels.csv')
n = len(df)
# x_df = pd.DataFrame(columns=df.columns.values)
# height = np.zeros(n)
# width = np.zeros(n)

# label = ['object']*n
# print(n)
for i in range(n):
# 	print(str(i)+"/"+str(n))
	img_name = df.loc[i, 'filename']
	image = cv2.imread("./large_images/"+img_name)
	cv2.imwrite("./images/"+img_name, image)
	# try:
	# 	if img_name is '1-1.png':
	# 		print(i)
	# 	img = cv2.imread("./images/"+str(img_name))
	# 	x_df.append(df[i])
	# 	# height[i], width[i] = img.shape[:2]
	# except:
	# 	pass
# print(x_df)
# x_df.to_csv('./data/train_labels.csv', index=False)
