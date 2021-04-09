from skimage import io
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth, AgglomerativeClustering
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from os import listdir
from os.path import isfile, join
from google.colab import drive
import pickle
import math
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats

drive.mount('/content/drive/')

# param: image (string)
# takes the image name (and location) as an input

# output: tuple
# tuple[0] = scale (and noise)
# tuple[1] = preprocessed image

def preprocess_image(image_name, med_blur):
  img = cv2.imread(image_name)
  color_im_2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  color_im_3 = cv2.cvtColor(color_im_2, cv2.COLOR_RGB2HSV)

  ## Show original image
  # plt.figure()
  # plt.imshow(color_im_2)
  # plt.title('Original Image')

  # Make numpy versions
  RGBna = np.array(color_im_2)
  HSVna = np.array(color_im_3)

  # Extract Hue
  H = HSVna[:,:,0]
  # Extract Value
  V = HSVna[:,:,2]

  # Find all pink pixels, i.e. where 180 < Hue < 340
  lo,hi = -20, 180
  # Rescale to 0-255, rather than 0-360 because we are using uint8
  lo = int((lo * 255) / 360)
  hi = int((hi * 255) / 360)
  notPink = np.where((H>lo) & (H<hi))

  notDark = np.where(V > 30)

  # Make all notPink pixels white in original image
  RGBna[notPink] = [255,255,255]

  gray_im_4 = cv2.cvtColor(RGBna, cv2.COLOR_RGB2GRAY)
  
  # gamma correction, blur, adaptive threshold, median blur
  gray_correct_2 = np.array(255 * (gray_im_4 / 255) ** 3.7 , dtype='uint8')
  blur = cv2.GaussianBlur(gray_correct_2,(17,17),1)
  ## Show Gaussian Blur
  # plt.figure()
  # plt.imshow(blur, cmap="gray")
  # plt.title('Gaussian Blur Applied To Grayscale')

  thresh = cv2.adaptiveThreshold(blur, 7050, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 111, 2)
  thresh = cv2.bitwise_not(thresh)
  ## Show Adaptive Threshold
  # plt.figure()
  # plt.imshow(thresh, cmap="gray")
  # plt.title('Adaptive Threshold Applied')

  median = cv2.medianBlur(thresh, med_blur)
  # plt.figure()
  # plt.imshow(median, cmap="gray")
  # plt.title('Median Filter Applied')

  return median


def extract_scale(img_name): ##if the height not in range +- 205 and 102 then search through array to find one in the height, if not print out name of message of picture
  img = cv2.imread(img_name)
  color_im_2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  color_im_3 = cv2.cvtColor(color_im_2, cv2.COLOR_RGB2HSV)
  RGBna2 = np.array(color_im_2)

  HSVna = np.array(color_im_3)
  V = HSVna[:,:,2]
  notDark = np.where(V > 7)
  RGBna2[notDark] = [255,255,255]

  scale_shown = cv2.cvtColor(RGBna2, cv2.COLOR_RGB2GRAY)

  scale_shown = np.array(255 * (scale_shown / 255) ** 3.7 , dtype='uint8')
  scale_shown = cv2.medianBlur(scale_shown, 11)
  scale_shown = cv2.bitwise_not(scale_shown)  ## invert black and white

  # plt.figure()
  # plt.imshow(scale_shown, cmap="gray")
  # plt.title('SCALE FUNCTION')

  output = cv2.connectedComponentsWithStats(scale_shown)
  num_labels = output[0]
  labels = output[1]
  stats = output[2]

  heights = []
  for i in range(num_labels):
    heights.append(stats[i, cv2.CC_STAT_HEIGHT])
  heights.sort()

  ## The scale will be the second tallest connected component (tallest being the whole background)
  scale_height = None
  for height in heights:
    if ((height >= 98 and height <= 105) or((height >= 202 and height <= 208))):
      scale_height = height  

  if scale_height == None:
    scale_height = 102

  ## Plot connected Components
  label_hue = np.uint8(179 * labels / np.max(labels))
  blank_ch = 255 * np.ones_like(label_hue)
  labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
  labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
  labeled_img[label_hue == 0] = 0

  plt.figure()
  plt.title('Scale Connect Components with Scale Height ' + str(scale_height))
  plt.imshow(labeled_img)
  plt.show()

  return scale_height


def initial_connected_components(median_img):
  ret, labels = cv2.connectedComponents(median_img)
  label_hue = np.uint8(179 * labels / np.max(labels))
  blank_ch = 255 * np.ones_like(label_hue)
  labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
  labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
  labeled_img[label_hue == 0] = 0

  # plt.figure()
  # plt.title('Objects counted:'+ str(ret-1))
  # plt.imshow(labeled_img)
  # plt.show()



## Store the width, height, and area of each vessel element
## Used this post to help (https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python)
def extract_data(blur_img, pxl_to_micro):
  output = cv2.connectedComponentsWithStats(blur_img)
  num_labels = output[0]
  labels = output[1]
  stats = output[2]
  centroids = output[3]  ## Maybe the centroids will be useful?

  vessels = []
  for i in range(num_labels):
    width = stats[i, cv2.CC_STAT_WIDTH] * pxl_to_micro
    height = stats[i, cv2.CC_STAT_HEIGHT] * pxl_to_micro
    area =  stats[i, cv2.CC_STAT_AREA] * pxl_to_micro
    left_start = stats[i, cv2.CC_STAT_LEFT]
    top_start = stats[i, cv2.CC_STAT_TOP]
    if (height > width):
      major_axis = height/2
      minor_axis = width/2
      center = (minor_axis,major_axis)
      foci_distance = math.sqrt(major_axis**2 - minor_axis**2)
      eccentricity = foci_distance / major_axis 
      # print(eccentricity)
      # foci_1 = (minor_axis,major_axis + foci_distance)
      # foci_2 = (minor_axis,major_axis - foci_distance)
    elif (width > height):
      major_axis = width/2
      minor_axis = height/2
      center = (major_axis,minor_axis)
      foci_distance = math.sqrt(major_axis**2 - minor_axis**2)
      eccentricity = foci_distance / major_axis 
      # print(eccentricity)

      # foci_1 = (major_axis + foci_distance,minor_axis)
      # foci_2 = (major_axis - foci_distance,minor_axis)
    else: 
      eccentricity = 0 

    vessels.append([width, height, area, left_start, top_start, eccentricity])

  vessel_df = pd.DataFrame(vessels, columns=['width', 'height', 'area', 'left_start_pxl', 'top_start_pxl','eccentricity'])
  
  return (vessel_df, labels)


## Perform outlier detection on both pixel location and pixel size of detected objects to remove the detected objects that aren't vessel elements
def drop_extras(vessel_df, min_area, max_area):
  outliers_indices = vessel_df.loc[(vessel_df['area'] > max_area) | (vessel_df['area'] < min_area) | (vessel_df['eccentricity'] > 0.90 )].index.tolist()
  print("Num outliers", len(outliers_indices))
  vessel_df = vessel_df.drop(outliers_indices, 0).reset_index(drop=True)
  print("Num objs left", len(vessel_df['area']))

  return vessel_df, outliers_indices


## Note: this function may take a bit of time to run
def remove_extras_image(labels, outliers_indices):
  ## Remove outliers from labels img by giving them the "1" label
  for outlier in outliers_indices:
    labels[labels == outlier] = 1
    
  ## Display the non-outliers
  label_hue = np.uint8(179 * labels / np.max(labels))
  blank_ch = 255 * np.ones_like(label_hue)
  labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
  labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
  labeled_img[label_hue == 0] = 0

  plt.figure()
  plt.imshow(labeled_img)
  plt.title("Vessel Elements")


## Function to calculate vessel density, by making a rectangle that
## encompasses all the vessels within
def calc_v_density(img_df, pxl_to_milli):
  left_start_col = img_df['left_start_pxl']
  top_start_col = img_df['top_start_pxl']

  rect_x = img_df['left_start_pxl'].min()
  rect_y = img_df['top_start_pxl'].min()

  rect_bottom_x = img_df['left_start_pxl'].max()
  rect_bottom_y = img_df['top_start_pxl'].max()

  # find the associated vessels with rect_bottom_x & rect_bottom_x
  max_index_left = left_start_col.idxmax()
  max_index_bottom = top_start_col.idxmax()

  rect_w = (rect_bottom_x + img_df['width'][max_index_left]*pxl_to_milli*1000) - rect_x
  rect_h = (rect_bottom_y + img_df['height'][max_index_bottom]*pxl_to_milli*1000) - rect_y

  # rect_area in mm^2
  rect_area = (rect_w * pxl_to_milli) * (rect_h * pxl_to_milli)
  img_df['vessel_density'] = img_df.shape[0] / rect_area
  calc_v_vulnerability_i(img_df)
  return img_df

## Function to calculate vessel vulnerability index
def calc_v_vulnerability_i(img_df):
  v_i = []
  for i in range(img_df.shape[0]):
    vessel_w = img_df['width'][i]
    vessel_h = img_df['height'][i]
    diameter = 0
    if (vessel_w > vessel_h):
      diameter = vessel_w * 2
    else:
      diameter = vessel_h * 2
    v_i.append(diameter / img_df['vessel_density'][i])
  img_df['vessel_vulnerability_index'] = v_i


## agglomerated clustering for automatic cluster detection of vessels

## sources:
## https://www.datanovia.com/en/lessons/agglomerative-hierarchical-clustering/
## https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
def detect_vessel_clustering(img_df):
  vessel_coor = img_df[['left_start_pxl', 'top_start_pxl']].copy()
  clustering = AgglomerativeClustering(distance_threshold = 70, n_clusters = None, compute_full_tree = True, linkage = 'single')
  c = clustering.fit_predict(vessel_coor)

  img_df['group_name'] = c
  calc_v_grouping_indices(img_df)

## calculating the vessel grouping indices
def calc_v_grouping_indices(img_df):
  num_vessels = img_df.shape[0]
  num_groupings = get_num_groups(img_df)
  num_solitary_vessels = num_vessels - num_groupings

  if (num_groupings != 0):
    v_g = num_vessels / num_groupings
    v_s = num_solitary_vessels / num_groupings
  else:
    v_g = 0
    v_s = 0

  f_vm = num_groupings / num_vessels
  img_df['vessel_grouping_index'] = v_g
  img_df['solitary_vessel_index'] = v_s
  img_df['vessel_multiple_fraction'] = f_vm

## Helper function for finding the number of vessel clusters
def get_num_groups(img_df):
  s = img_df['group_name'].value_counts()
  s = s[s>1]
  return len(s)

## Helper function for plotting the autogenerated clusters
## returns df of just the groups
def plot_just_groups(img_df):
  img_df_2 = img_df.copy()
  s = img_df_2['group_name'].value_counts()
  s = s[s>1]

  img_df_2 = img_df_2[img_df_2['group_name'].isin(s.index)]
  plt.figure() 
  scatter = plt.scatter(img_df_2['left_start_pxl'], img_df_2['top_start_pxl'],  
              c = img_df_2['group_name'], cmap ='rainbow') 
  plt.title('group location')
  ax = scatter.axes
  ax.invert_yaxis()
  plt.show(scatter)
  return img_df_2

## Driver function of the entire image analysis process
def analyze_img(img_name, med_blur, min_area, max_area):
  median_img = preprocess_image(img_name, med_blur)

  scale_height = extract_scale(img_name)
  ## Convert pixels to micrometers
  pxl_to_micro = scale_height / 100
  ## Convert pixels to millimeters
  pxl_to_milli = pxl_to_micro / 1000

  ## Show initial connected components
  initial_connected_components(median_img)

  ## Extract data from image and put into Dataframe
  data = extract_data(median_img, pxl_to_micro)
  vessel_df = data[0]
  labels = data[1]
  
  ## Drop the extra detected connected components
  data = drop_extras(vessel_df, min_area, max_area)
  vessel_df = data[0]
  outlier_indices = data[1]

  ## Show the connected components with outliers removed
  remove_extras_image(labels, outlier_indices)

  ## Adding vessel info
  calc_v_density(vessel_df, pxl_to_milli)
  detect_vessel_clustering(vessel_df)
  plot_just_groups(vessel_df)

  print("Total elements:", len(vessel_df))
  
  return vessel_df
