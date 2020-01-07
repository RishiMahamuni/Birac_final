# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:42:10 2019

@author: BharatAgri Rishikeesh

"""

from __future__ import print_function
import importlib
from distutils.version import LooseVersion
from configparser import ConfigParser
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
import random 
import gc
import glob
from tqdm import tqdm
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
#from torchsummary import summary
#import pytorch



####### Load_images starts ##########
def load_image():
    '''
    Doctstring:
        Loading all 4 classes of images for disease severity stage
    '''
    healthy = []
    s0 = []
    s1 = []
    s2 = []
    paths = config_parse()
    for i in paths:
        if 'healthy' in i:           
            data_path = os.path.join(i,'*.*')
            files = glob.glob(data_path)
            for f1 in files:
                img = cv2.imread(f1)
                healthy.append(img)
        elif 's0' in i:
            data_path = os.path.join(i,'*.*')
            files = glob.glob(data_path)
            for f1 in files:
                img = cv2.imread(f1)
                s0.append(img)
        elif 's1' in i:
            data_path = os.path.join(i,'*.*')
            files = glob.glob(data_path)
            for f1 in files:
                img = cv2.imread(f1)
                s1.append(img)
        else:
            data_path = os.path.join(i,'*.*')
            files = glob.glob(data_path)
            for f1 in files:
                img = cv2.imread(f1)
                s2.append(img)
            
    '''            
    print ("healthy category count = %d" %(len(healthy))) 
    print ("S0 category count = %d" %(len(s0))) 
    print ("S1 category count = %d" %(len(s1)))
    print ("S2 category count = %d" %(len(s2)))'''
    
    return healthy,s0,s1,s2    
    
######### Load Imnages Ends ############

    
############ configuration starts ##############
def config_parse():
    """ 
    Comfigurations of all the things  
    
    parameters: no parameteris,
    
    
    returns:returns the path of image data 
    
    """
    
    config = ConfigParser()
    config.read('config_file.ini')
   # print(config.sections())
    #for key in config['path']:
       #print(key)
    healthy_class = config.get('image_path_exp1_comb1','healthy')
    s0_class = config.get('image_path_exp1_comb1','s0')
    s1_class = config.get('image_path_exp1_comb1','s1')
    s2_class = config.get('image_path_exp1_comb1','s2')
    return healthy_class,s0_class,s1_class,s2_class
    
############ configuration ends ############



######## adjust Bright starts #########
def adjust_bright():
    ''' 
    Docstring : adjust the brightness of the dark images .
    those images are usually the images with band000+band001+band002+band003
    input : 4 classes of images healthy , s0, s1, s2 
    output : 4 classes with adjusted brightness
    '''
    pass
     
####### adjust Brightv Ends #########
 
   
    




######## histogram_equalization starts#########
def histogram_equalization(images):
    healthy = images[0]
    s0 = images[1]
    s1 = images[2]
    s2 = images[3]
    '''
    print(len(healthy) )
    print(len(s0) )
    print(len(s1) )
    print(len(s2) )
    '''
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) method does grid wise group in the image and then performs the histogram correction
     
    for i in range(len(healthy)):
        grayimg = cv2.cvtColor(healthy[i], cv2.COLOR_BGR2GRAY)
        healthy[i] = cv2.equalizeHist(grayimg)
    
    for i in range(len(s0)):
        grayimg = cv2.cvtColor(s0[i], cv2.COLOR_BGR2GRAY)
        s0[i] = cv2.equalizeHist(grayimg)
        
    for i in range(len(s1)):
        grayimg = cv2.cvtColor(s1[i], cv2.COLOR_BGR2GRAY)
        s1[i] = cv2.equalizeHist(grayimg)
        
    for i in range(len(s2)):
        grayimg = cv2.cvtColor(s2[i], cv2.COLOR_BGR2GRAY)
        #s2[i] =clahe.apply(grayimg)
        s2[i] = cv2.equalizeHist(grayimg)
        
    '''print ("healthy category count = %d" %(len(healthy))) 
    print ("S0 category count = %d" %(len(s0))) 
    print ("S1 category count = %d" %(len(s1)))
    print ("S2 category count = %d" %(len(s2)))'''   
    
    

    
    '''cv2.imshow('image',healthy[120])   
    cv2.waitKey(0)
    cv2.imshow('image',s0[50])   
    cv2.waitKey(0)
    cv2.imshow('image',s1[50])   
    cv2.waitKey(0)
    cv2.imshow('image',s2[50])   
    cv2.waitKey(0)'''
    
    print('histogram success')   
    return healthy,s0,s1,s2 
    
    #return images 
    #print('histogram success')
########histogram_equalization ends ##########   




######## image denoise starts ###########    
def image_denoise(healthy,s0,s1,s2):
    '''
    Docstring:
        image denoising using the gaussian filter 
        
        input:image data
        
        output : denoise data
    '''
    for i in range(len(healthy)):
        healthy[i] = cv2.GaussianBlur(healthy[i],(5,5),0)
    
    for i in range(len(s0)):
        s0[i] = cv2.GaussianBlur(s0[i],(5,5),0)
        
    for i in range(len(s1)):
       s1[i] = cv2.GaussianBlur(s1[i],(5,5),0)
        
    for i in range(len(s2)):
        s2[i] = cv2.GaussianBlur(s2[i],(5,5),0)
        
    '''cv2.imshow('image',healthy[120])   
    cv2.waitKey(0)
    cv2.imshow('image',s0[50])   
    cv2.waitKey(0)
    cv2.imshow('image',s1[50])   
    cv2.waitKey(0)
    cv2.imshow('image',s2[50])   
    cv2.waitKey(0)'''
    print('denoise success')
    return healthy,s0,s1,s2          
########## image denoise ends ############  
  
    
    
##### show_image starts#########
def show_image(images):
    cv2.imshow('image',images[0][100])   
    cv2.waitKey(0)
    
    
###### show image ends ########



######## image_augmentation starts #######

def image_augmenting(healthy,s0,s1,s2):
    
    '''
    Docstring : image augmentation using pytorch library
    
    input : images of all 4 classes
    '''
    train_x = np.array(s0)
    train_x1 = np.array(s1)
    train_x2 = np.array(s2)
    s0 = []
    s1 = []
    s2 = []
    for i in tqdm(range(train_x.shape[0])):
        s0.append(train_x[i])
        s0.append(rotate(train_x[i], angle=45, mode = 'wrap'))
        s0.append(np.fliplr(train_x[i]))
        s0.append(np.flipud(train_x[i]))
        s0.append(random_noise(train_x[i],var=0.2**2))
    #print(len(final_s2))
    s0 = np.array(s0)
    print(len(s0))
    
    for i in tqdm(range(train_x1.shape[0])):
        s1.append(train_x1[i])
        s1.append(rotate(train_x1[i], angle=45, mode = 'wrap'))
        s1.append(np.fliplr(train_x1[i]))
        s1.append(np.flipud(train_x1[i]))
        s1.append(random_noise(train_x1[i],var=0.2**2))
    #print(len(final_s2))
    s1 = np.array(s1)
    print(len(s1))
    
    for i in tqdm(range(train_x2.shape[0])):
        s2.append(train_x2[i])
        s2.append(rotate(train_x2[i], angle=45, mode = 'wrap'))
        s2.append(np.fliplr(train_x2[i]))
        s2.append(np.flipud(train_x2[i]))
        s2.append(random_noise(train_x2[i],var=0.2**2))
    #print(len(final_s2))
    s2 = np.array(s2)
    print(len(s2))
    print('image augmenting done')
             
######## image_augmnetation ends ########


######### test starts#########
def test(s2):
    
    '''data_s2 = r'D:\Project BIG\classifiers\ri_leaf\s2'
    data_path = os.path.join(data_s2,'*.*')
    files = glob.glob(data_path)
    s2 = []
    for f1 in files:        
        img = cv2.imread(f1)
        img = img/255
        s2.append(img)'''
        
    train_x = np.array(s2)
    final_s2 = []
    for i in tqdm(range(train_x.shape[0])):
        final_s2.append(train_x[i])
        final_s2.append(rotate(train_x[i], angle=45, mode = 'wrap'))
        final_s2.append(np.fliplr(train_x[i]))
        final_s2.append(np.flipud(train_x[i]))
        final_s2.append(random_noise(train_x[i],var=0.2**2))
    #print(len(final_s2))
    final_train= []
    final_train = np.array(final_s2)
    print(len(final_train))
    
    '''
    #Visualising the data 
    fig,ax = plt.subplots(nrows=1,ncols=5,figsize=(20,20))
    for i in range(5):
        ax[i].imshow(final_train[i+30])
        ax[i].axis('off')
'''
######### test ends ########



######preprocessing starts ##############    
def preprocessing(): 
    '''
    Docstring :
        all preprcocessing stages covered inside this function:
            1. loading of images 
            2. brightness adjustment 
            3. histogram equalisation
            4. Noise reduction
    '''
    
    healthy = []
    s0 = [] 
    s1 = []
    s2 = []
    print('In preprocessing block')
    images = load_image()
    #adjust_bright()
    healthy,s0,s1,s2 = histogram_equalization(images)

    healthy,s0,s1,s2 = image_denoise(healthy,s0,s1,s2)
    
    image_augmenting(healthy,s0,s1,s2)   
    #test(s2)
    #print('denoise completed')
    #show_image(images)
    
####### preprocessing ends #################





###############Dependancy Check starts###############
def dependancy_check():
    """ 
         checking the dependancies of packages with version 
         
         parameter: no parameters
         
         returns: True: no dependancy
         false: if any dependancy
       
    """
   
    # check that all packages are installed (see requirements.txt file)
    required_packages = {'jupyter', 
                     'numpy',
                     'matplotlib',
                     'ipywidgets',
                     'scipy',
                     'pandas',
                     'random',
                     'gc',
                     'glob'
                    }

    problem_packages = list()
    # Iterate over the required packages: If the package is not installed
    # ignore the exception. 
    for package in required_packages:
        
        try:
            p = importlib.import_module(package)        
        except ImportError:
           
            problem_packages.append(package)
    
    if len(problem_packages) is 0:
        #print('All is well.')
        return 0;
    else:
        #print('The following packages are required but not installed: ' \
          #+ ', '.join(problem_packages))
        return problem_packages;

#################### dependancy check ends ################
        




######## main function Starts ############
def main():
    req_package = dependancy_check()
    if req_package == 0:
        preprocessing()
    else:
        #load_images()
        print('The following packages are required but not installed: ' \
          + ' , '.join(req_package))
           
############m main function ends #############3
   


    
if __name__ == '__main__':
    #config_parse()
    main()
   #preprocessing()

