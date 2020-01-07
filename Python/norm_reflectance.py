# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:33:59 2019

@author: BharatAgri

@normalising the reflectance data and crate a sheet for further preprocessing and 

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
from os import walk




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
    path = config.get('path', 'reflectance_file')
    return path
############ configuration ends ############

    
        
####read_files_starts####
def read_files():
    
    '''
    Reading the file names from the folder 
    
    parameter: none
    
    Returns: list of file_names
    
    '''
    
    path = config_parse()
    #path = 'C:\\Users\\BharatAgri\\Birac\\SR_E1\\'
    #files = []
    
    ref_files = []
    
    for (dirpath, dirnames, filenames) in walk(path):
        
         ref_files.extend(filenames)
         break
     
    return ref_files,path
    
###read_files_ends######      
    
 
    

######preprocessing_starts############
        
def preprocessing():
    print('preprocessin block starts')
    ref_files,path = read_files()
    
    #print('files_read_successfully')
    #print(ref_files[0])
    
    sample_loc = []
    new_columns = []
    #reading a excel file 
    df = pd.read_csv(path+ref_files[0],skiprows=4)
    
    #extracting the column name except the wavelength which are reflectance sample locations 
    for c in df.columns:
        if 'Wavelength (nm)' in c:
            continue
        else:
            sample_loc.append(c)
    
    #writing location data to separate sheet
    #lengh_sam = len(sample_loc)
    
    #new_columns = [new_columns for i in range(len(sampple_loc)) ref_files[0].strip('.csv')+'_'+str(i+1)]
    
    for i in range(len(sample_loc)):
        new_columns.append(ref_files[0].strip('.csv')+'_'+str(i+1))
        
    print(len(sample_loc))
    print(len(new_columns))
    
    for s in df.columns:
         if 'Wavelength (nm)' in s:
            continue
         elif len(sample_loc) == len(new_columns):
                  
        
            #df.rename(columns = {}, inplace=True)
    
    #extracting the column names except wavelength (nm)
    #sam_location = [ df.columns for x in df.columns if ]
   # print(sam_location)
     #print('preprocessing block ends')
######preprocessing_ends############      
    
    

######## main function Starts ############
def main( ):
    
    #print("data science ")
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
   main()
   #preprocessing()
