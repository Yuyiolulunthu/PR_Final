import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
from PIL import Image
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor

def extract_feature(base_dir, output_csv, flag):
    # read in the image
    emotions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    data = []
    for label in emotions:
        label_dir = os.path.join(base_dir, label)
        for file in tqdm(os.listdir(label_dir), desc=f'Processing {label}'):
            filepath = os.path.join(base_dir, label, file)
            try:
                # in format of 48 * 48 or 100 * 100
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if (flag == 'emlbp'):
                    feature = extract_EMLBP(image)
                elif (flag == 'hog'):
                    feature = extract_HOG(image)
                elif (flag == 'gabor'):
                    feature = extract_Gabor(image)
                if (file == 'test_2389_aligned.jpg'):
                    cv2.imshow('test', feature)
                feature_str = ' '.join(map(str, feature))
                row = [file, feature_str, label]
                data.append(row)
            except:
                print(f'no {filepath} in the csv file')
            
    # write to the output csv file
    df = pd.DataFrame(data, columns=['file', 'feature', 'label'])
    df.to_csv(output_csv, index=False)
                   
def extract_EMLBP(image, P_R_list=[(8,1), (16,2), (24,3)]):
    feature = []
    for P, R in P_R_list:
        lbp_feature = local_binary_pattern(image, P, R, method='uniform')
        hist, _ = np.histogram(lbp_feature.ravel(), bins=np.arange(0, P + 3), density=True)
        feature.extend(hist)
    return np.array(feature).astype(np.float32)

def extract_HOG(image, orientations=9, pixel_per_cell=(8,8), cells_per_block=(2,2)):
    hog_feature = hog(image, orientations=orientations, pixels_per_cell=pixel_per_cell, 
                      cells_per_block=cells_per_block)
    return hog_feature.astype(np.float32)

def extract_Gabor(image, frequencies=[0.1, 0.2, 0.3], thetas=[0, np.pi/4, np.pi/2]):
        feature = []
        for frequency in frequencies:
            for theta in thetas:
                filt_r, _ = gabor(image, frequency=frequency, theta=theta)
                feature.append(filt_r.mean())
                feature.append(filt_r.var())
        return np.array(feature).astype(np.float32)
    
# test_image = cv2.imread('FER2013\\train\\happy\\Training_99707061.jpg', cv2.IMREAD_GRAYSCALE)   
# print(test_image.shape)   
# test_feature = extract_EMLBP(test_image)
# print(test_feature.shape)
# test_feature = extract_HOG(test_image)
# print(test_feature.shape)
# test_feature = extract_Gabor(test_image)
# print(test_feature.shape)

FER_PLUS_train = 'FER_PLUS/train'
FER_PLUS_test = 'FER_PLUS/test'
RAFDB_train = 'RAF-DB/DATASET/train'
RAFDB_test = 'RAF-DB/DATASET/test'

extract_feature(FER_PLUS_train, 'fer_train_emlbp.csv', 'emlbp')
extract_feature(FER_PLUS_test, 'fer_test_emlbp.csv', 'emlbp')
extract_feature(RAFDB_train, 'rafdb_train_emlbp.csv', 'emlbp')
extract_feature(RAFDB_test, 'rafdb_test_emlbp.csv', 'emlbp')
extract_feature(FER_PLUS_train, 'fer_train_hog.csv', 'hog')
extract_feature(FER_PLUS_test, 'fer_test_hog.csv', 'hog')
extract_feature(RAFDB_train, 'rafdb_train_hog.csv', 'hog')
extract_feature(RAFDB_test, 'rafdb_test_hog.csv', 'hog')
# extract_feature(FER_PLUS_train, 'fer_train_gabor.csv', 'gabor')
# extract_feature(FER_PLUS_test, 'fer_test_gabor.csv', 'gabor')
# extract_feature(RAFDB_train, 'rafdb_train_gabor.csv', 'gabor')
# extract_feature(RAFDB_test, 'rafdb_test_gabor.csv', 'gabor')
