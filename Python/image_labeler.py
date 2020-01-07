# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:04:54 2020

@author: BharatAgri Rishikeesh

@reading images from the folder and listing their names in the .csv file with class of image

"""

from __future__ import print_function
import importlib
from distutils.version import LooseVersion
from configparser import ConfigParser
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os
import xlrd
import re
import os


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
                     'pandas'
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
    sample = config.get('image_path_exp1_comb1','sample')
    #return healthy_class,s0_class,s1_class,s2_class
    return sample
############ configuration ends ############
   
    

###### rename_image starts #########
def rename_image():
    ''' 
    docstring : rename all the images in the folder 
    
    '''
    dir_path = config_parse()
    
    i=0 
    
    for filename in os.listdir(dir_path):
        print(filename)
        dist = "s2" + str(i) + ".bmp"
        src = filename
        print(src)
        os.rename(src,dist)
        i += 1
   

###### rename_image ends ##########
    
    


######### image labeler Starts ########    
def image_labeler():
    ''' 
    Docstring :reades all the images from 4 classes rename it
    ii. put all the renamed images in single folder 
    iii. creates the .csv file for image names and class label
    
    '''
    rename_image()   
    
       
####### image labeler ends ##########
    
    
    
    
######## main function Starts ############
def main():
    
    print('in main')
    check = dependancy_check()
    if check == 0:
        image_labeler()
    else:
        #load_images()
        print('The following packages are required but not installed: ' \
          + ', '.join(check))
           
############m manin function ends #############3
   
    
    
    
    
if __name__ == '__main__':
    #config_parse()
    main()
   #preprocessing()

    


