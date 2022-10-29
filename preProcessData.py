# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 10:53:31 2022

@author: dirty
"""

import os , sys
from pathlib import Path
import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder ,StandardScaler,LabelEncoder
from PIL import Image
from tqdm import tqdm

def imageResize(trainOrTest,fileNameList) :
    
    origin_path = f"../data/{trainOrTest}_images/"
    processed_path = f'../data/{trainOrTest}_images_processed/'
    
    for fileName in tqdm(fileNameList) :
        img = Image.open( origin_path + fileName)
        new_img = img.resize((512, 512))
        new_img.save( processed_path + fileName)


def preProcessData(trainOrTest ,df) :
    
    imageResize(trainOrTest,df['image'])
    
    
    
    return df 
    
def main():
    path = '../data/'
    train_df = pd.read_csv( path + 'train.csv')
    test_df = pd.read_csv( path + 'test.csv')
    
    train_processed = preProcessData('train',train_df)
    test_processed = preProcessData('test',test_df)
    
    
    # save 
    train_processed.to_csv(path + 'train_processed.csv',index=False)
    test_processed.to_csv(path + 'test_processed.csv',index=False)
    
    
    
if __name__ == "__main__":
    Path("../data/train_images_processed").mkdir(parents=True, exist_ok=True)
    Path("../data/test_images_processed").mkdir(parents=True, exist_ok=True)
    main()