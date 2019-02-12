import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append("..")
from object_detection.utils import ops as utils_ops


get_ipython().run_line_magic('matplotlib', 'inline')


from utils import label_map_util
from utils import visualization_utils as vis_util


# In[4]:


# What model to download.
MODEL_NAME = 'trained_model'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('model', 'object-detection.pbtxt')


# In[5]:


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# In[6]:


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# In[7]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# In[8]:


df = pd.read_csv('large_data/OriginalData/final_test_labels.csv')
if 'image_name' in df.columns:
    df.rename(columns={'image_name': 'filename'}, inplace=True)


# In[9]:


image_name = df['filename']
# image_name.head()


# In[10]:


# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'images/'
# PATH_TO_TEST_IMAGES_DIR = 'test_images'

# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]
# TEST_IMAGE_NAMES = ['JPEG_20160627_165046_100057407407.png']
# TEST_IMAGE_NAMES = ['JPEG_20161126_172142_1000140435770.png',
# 'JPEG_20160526_143251_1000818472906.png',
# '1473486851140DeeplearnS11925.png',
# '147772288053420161028_160034.png',
# 'JPEG_20160624_153535_1000651561441.png',
# 'JPEG_20160810_164140_1000336896328.png',
# '1469620012315JPEG_20160625_112201_1000107496509.png',
# 'JPEG_20160701_110715_1000156257054.png',
# '1473317444895DeeplearnS1212.png',
# '1470739817043DSC_0180.png',
# 'JPEG_20160711_144858_100092253192.png',
# 'JPEG_20160513_161311_1000637195107.png',
# '1468495287839DSC_0181.png',
# '1470746335192DSC_0249.png',
# '1480333088202_R2A3481.png',
# 'JPEG_20160821_153314_1000632181893.png',
# '147772067540420161028_163422.png',
# '1473750918573IMG_1012.png',
# '1473662453571DeeplearnS11523.png',
# '147771763106320161028_170936.png',
# 'JPEG_20160724_121802_1000231980248.png',
# 'JPEG_20160524_153045_1000595292696.png',
# 'JPEG_20161028_151604_1000660130717.png',
# '1480961744808_R2A2451.png',
# 'JPEG_20160831_162216_1000209602261.png',
# '1480067308546IMG_0167.png',
# 'JPEG_20160709_120153_1000240917049.png',
# '147772354371220161028_162011.png',
# '147772253973920161028_160310.png',
# 'JPEG_20161129_172959_1000748223678.png',
# 'JPEG_20161202_172017_1000911286173.png',
# '147444571185211470041583074-Roadster-Black-Casual-Shirt-991470041582862-2.png',
# '1472629285644s-l160017.png',
# 'JPEG_20160810_165829_1000395817645.png',
# '1474724303636DSC08136.png',
# '1474703063773DSC07138.png',
# 'JPEG_20160625_121137_1000661814045.png',
# '1474724221565DSC08146.png',
# 'JPEG_20160830_113822_1000234788155.png',
# 'JPEG_20161118_185140_1000881625779.png',
# '147754355690420161026_134224.png',
# 'JPEG_20161126_174910_1000380178382.png',
# 'JPEG_20160624_160048_1000713198675.png',
# '1472628515267w_lombok_black_patent_01.png',
# '1473750905899IMG_1010.png',
# 'JPEG_20160927_141021_1000405439828.png',
# 'JPEG_20160702_111718_1000405902246.png',
# '1474723417304DSC07767.png',
# 'JPEG_20160709_120514_1000165058605.png',
# '1474713548759DSC07469.png',
# '1475150409042DSC00480.png',
# 'JPEG_20161007_140052_1000349517635.png',
# '1474637826496DSC06440.png',
# 'JPEG_20161128_142037_1000410064030.png']

# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.png'.format(i)) for i in range(18, 22) ]

# TEST_IMAGE_PATHS = []
# for i in TEST_IMAGE_NAMES:
#     TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, i))

TEST_IMAGE_PATHS = PATH_TO_TEST_IMAGES_DIR +'/'+ image_name

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# In[11]:


# image_name.head()


# In[12]:


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


# In[13]:


# for image_path in TEST_IMAGE_PATHS:
#   image = Image.open(image_path)
#   # the array based representation of the image will be used later in order to prepare the
#   # result image with boxes and labels on it.
#   image_np = load_image_into_numpy_array(image)
#   # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#   image_np_expanded = np.expand_dims(image_np, axis=0)
#   # Actual detection.
#   output_dict = run_inference_for_single_image(image_np, detection_graph)
#   # Visualization of the results of a detection.
#   vis_util.visualize_boxes_and_labels_on_image_array(
#       image_np,
#       output_dict['detection_boxes'],
#       output_dict['detection_classes'],
#       output_dict['detection_scores'],
#       category_index,
#       instance_masks=output_dict.get('detection_masks'),
#       use_normalized_coordinates=True,
#       line_thickness=8,
#       max_boxes_to_draw=1,
#       min_score_thresh=0.50)
#   plt.figure(figsize=IMAGE_SIZE)
#   plt.imshow(image_np)


# In[15]:


from time import time
start = time()


# In[19]:


for img_indx in range(len(TEST_IMAGE_PATHS)):
    try:
        image_path = TEST_IMAGE_PATHS[img_indx]
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        #   print(output_dict['detection_boxes'][0])
        #   print(output_dict['detection_scores'][0])
        ymin, xmin, ymax, xmax = output_dict['detection_boxes'][0]
        image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
        im_width, im_height = image_pil.size
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
        left = int(left)
        right = int(right)
        top = int(top)
        bottom = int(bottom)
        x1 = left
        x2 = right
        y1 = top
        y2 = bottom
#         print(x1, x2, y1, y2)
        df.loc[img_indx, 'x1'] = int(x1)
        df.loc[img_indx, 'x2'] = int(x2)
        df.loc[img_indx, 'y1'] = int(y1)
        df.loc[img_indx, 'y2'] = int(y2)

        if img_indx%50 is 0:
            print(str(img_indx)+'/'+str(len(TEST_IMAGE_PATHS)))
    #     print(x1, x2, y1, y2)
    except:
        print("Error in image "+str(img_indx))
        df.loc[img_indx, 'x1'] = 95
        df.loc[img_indx, 'x2'] = 593
        df.loc[img_indx, 'y1'] = 180
        df.loc[img_indx, 'y2'] = 307
        pass
end = time()
print(end-start)


# In[18]:


# df.head()
df.rename(columns={'filename': 'image_name', 'xmin': 'x1', 'xmax': 'x2', 'ymin': 'y1', 'ymax': 'y2'}, inplace=True)
df.to_csv('./data/test_labeled.csv', index=False)


# In[37]:


# import cv2
# # indx = 6170
# # x1, y1 = df.loc[indx, 'x1'], df.loc[indx, 'y1']
# # x2, y2 = df.loc[indx, 'x2'], df.loc[indx, 'y2']
# # image = cv2.imread('large_images/'+df.loc[indx, 'filename'])
# x1, y1 = 95, 180
# x2, y2 = 593, 307
# image = cv2.imread('large_images/JPEG_20160627_165046_100057407407.png')
# image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
# cv2.imshow("Localisation", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

