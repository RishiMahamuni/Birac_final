# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:07:27 2020

@author: BharatAgri 

@basic structure python file

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
    """ configures all the things 
    
    parameters: no parameteris,
    
    
    returns: 1.healthy_filpath,
    2.inoculated_filepath
    
    """
    
    config = ConfigParser()
    config.read('config_file.ini')
   # print(config.sections())
    #for key in config['path']:
       #print(key)
    healthy_ds = config.get('path', 'reflectance_data_healthy')
    inoculated_ds = config.get('path', 'reflectance_data_inoculated')
    return healthy_ds,inoculated_ds
    
############ configuration ends ############
    
    
    
######## main function Starts ############
def main():
    
    print('in main')
    check = dependancy_check()
    if check == 0:
        preprocessing()
    else:
        #load_images()
        print('The following packages are required but not installed: ' \
          + ', '.join(check))
           
############m manin function ends #############3
   
    
    
    
    
if __name__ == '__main__':
    #config_parse()
  # main()
   preprocessing()

    
    

