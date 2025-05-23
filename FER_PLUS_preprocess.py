# In this file, simply preprocess the data of FER2013 and FER+ into the form of csv,
# with each image has 48 * 48 = 2304 pixels. FER2013 has single label, and FER+ has vector label

# original: FER2013 in directory of 7 emotions, each image is .jpg file
# original: FER+ in directory of 8 emotions(7 + contempt), each image is .png file
# a reference to the vector label of FER PLUS is fer2013new.csv

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

FER2013_train = 'FER2013/train'
FER2013_test = 'FER2013/test'
FER_PLUS_train = 'FER_PLUS/train'
FER_PLUS_test = 'FER_PLUS/test'

RAFDB_train = 'RAF-DB/DATASET/train'
RAFDB_test = 'RAF-DB/DATASET/test'



pixel_data = {}

fer2013new_df = pd.read_csv('FER_PLUS/fer2013new.csv')
fer2013new_df = fer2013new_df[fer2013new_df['Image name'].notna() & (fer2013new_df['Image name'] != '')]
fer2013new_df = fer2013new_df.set_index('Image name')


# read in the data from FER_PLUS
def image2csv(base_dir):
    emotions = os.listdir(FER_PLUS_train)
    for labels in emotions:
        label_dir = os.path.join(base_dir, labels)
        for file in tqdm(os.listdir(label_dir), desc=f'Processing {labels}'):
            if file in fer2013new_df.index:
                filepath = os.path.join(base_dir, labels, file)
                try:
                    image = Image.open(filepath).convert('L')
                    pixel_array = np.array(image).flatten()
                    pixel_str = ' '.join(map(str, pixel_array))
                    pixel_data[file] = pixel_str
                except:
                    print(f'no {filepath} in the csv file')
    
    # pixel_df = pd.DataFrame.from_dict(pixel_data, orient='index', columns=['pixels'])
    # pixel_df.index.name = 'Image name'
    # combined_df = fer2013new_df.join(pixel_df)
    # combined_df.to_csv('fer_plus.csv')
    fer2013new_df['pixels'] = fer2013new_df.index.map(lambda name: pixel_data.get(name, None))
    fer2013new_df.reset_index().to_csv('fer_plus.csv', index=False)

    
# image2csv(FER_PLUS_train)
# image2csv(FER_PLUS_test)        
            
