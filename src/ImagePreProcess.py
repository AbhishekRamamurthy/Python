'''
Created on Mar 28, 2018

@author: abhishek
'''

import cv2
import re 
import os
import pickle
#import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from skimage.transform.radon_transform import radon

'''
Types = {
    "Type1" : [1,0,0,0,0,0,0,0],
    "Type2" : [0,1,0,0,0,0,0,0],
    "Type3" : [0,0,1,0,0,0,0,0],
    "Type4" : [0,0,0,1,0,0,0,0],
    "Type5" : [0,0,0,0,1,0,0,0],
    "Type6" : [0,0,0,0,0,1,0,0],
    "Type7" : [0,0,0,0,0,0,1,0],
    "Type8" : [0,0,0,0,0,0,0,1]
    }
'''

Types = {
    "Type1" : 1,
    "Type2" : 2,
    "Type3" : 3,
    "Type4" : 4,
    "Type5" : 5,
    "Type6" : 6,
    "Type7" : 7,
    "Type8" : 8
    }

def Create_Labels_And_Features():
    pass

def PreProcess(path):
    pattern_out = re.compile("^Type.*$")
    pattern_in = re.compile("^Sub.*")
    pattern_image = re.compile("^.*.png$")
    features_dict = defaultdict(list)
    train =[]
    test = []

    for o_dir in os.listdir(path):
        
        if(pattern_out.match(o_dir)):
            for i_dir in os.listdir(path+"/"+o_dir):
                
                if pattern_in.match(i_dir):
                    for image in os.listdir(path+"/"+o_dir+"/"+i_dir):
                        if pattern_image.match(image):
                            matrix = cv2.imread(path+"/"+o_dir+"/"+i_dir+"/"+image)
                            cropped = matrix[208:277,81:576]
                            cropped = np.asarray(cropped,dtype='f')
                            cropped = np.ndarray.flatten(cropped)
                            features_dict[o_dir].append(cropped)
    
    keys = features_dict.keys()
    for key in keys:
        features = []
        temp = features_dict[key]
        for matrix in temp:
            matrix = np.ndarray.flatten(matrix)
            features.append([matrix,Types[key]])
        #features = np.array(features)
        if(len(temp) > 100):
            testing_size = int(0.1*len(temp))
            train.append(list(features[0:-testing_size]))
            test.append(list(features[-testing_size:]))
        else :
            testing_size = int(0.4*len(temp))
            train.append(list(features[0:-testing_size]))
            test.append(list(features[-testing_size:]))
    
    random.shuffle(train)
    random.shuffle(test)
    
    train_x = [item[0] for item in train[0]]
    train_y = [item[1] for item in train[0]]
    test_x = [item[0] for item in test[0]]
    test_y = [item[1] for item in test[0]]
        
    with open(os.getcwd()+"/Image_preprocess.pickle",'wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)
    #return train_x,train_y,test_x,test_y
    
