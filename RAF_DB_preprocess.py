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

RAFDB_train = 'RAF-DB/DATASET/train'
RAFDB_test = 'RAF-DB/DATASET/test'


pixel_data = {}

# fer2013new_df = pd.read_csv('.csv')
# fer2013new_df = fer2013new_df[fer2013new_df['Image name'].notna() & (fer2013new_df['Image name'] != '')]
# fer2013new_df = fer2013new_df.set_index('image')


# read in the data from FER_PLUS
def image2csv(base_dir, flag):
    label_csv = os.path.join(base_dir, flag + '_labels.csv')
    rafdb_df = pd.read_csv(label_csv)
    rafdb_df = rafdb_df.set_index('image')
    emotions = os.listdir(base_dir)
    for labels in emotions:
        label_dir = os.path.join(base_dir, labels)
        if os.path.isdir(label_dir):
            for file in tqdm(os.listdir(label_dir), desc=f'Processing {labels}'):
                if file in rafdb_df.index:
                    filepath = os.path.join(base_dir, labels, file)
                    try:
                        image = Image.open(filepath).convert('L')
                        pixel_array = np.array(image).flatten()
                        pixel_str = ' '.join(map(str, pixel_array))
                        pixel_data[file] = pixel_str
                    except:
                        print(f'no {filepath} in the csv file')
    

    rafdb_df['pixels'] = rafdb_df.index.map(lambda name: pixel_data.get(name, None))
    csv_filename = 'rafdb' + flag + '.csv'
    rafdb_df.reset_index().to_csv(csv_filename, index=False)

    
image2csv(RAFDB_train, 'train')
image2csv(RAFDB_test, 'test')        
            
