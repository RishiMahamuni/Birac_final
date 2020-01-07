# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:52:38 2019

@author: BharatAgri

@detail setup and the preprocessing of the numerical data 
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
    
    




########### get_sheet_name_starts ###########
'''
using this iun the stage where will have so many sheets and need to reduce the loading time
basically this is optimised sheet loading reducer
def get_sheet_details(file_path):
    sheets = []
    file_name = os.path.splitext(os.path.split(file_path)[-1])[0]
    # Making a temporary directory with the file name
    directory_to_extract_to = os.path.join(settings.MEDIA_ROOT, file_name)
    os.mkdir(directory_to_extract_to)

    # Extracting the xls file as it is just a zip file
    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()

    # Open the workbook.xml which is very light and only has meta data, get sheets from it
    path_to_workbook = os.path.join(directory_to_extract_to, 'xl', 'workbook.xml')
    with open(path_to_workbook, 'r') as f:
        xml = f.read()
        dictionary = xmltodict.parse(xml)
        for sheet in dictionary['workbook']['sheets']['sheet']:
            sheet_details = {
                'id': sheet['sheetId'], # can be @sheetId for some versions
                'name': sheet['name'] # can be @name
            }
            sheets.append(sheet_details)

    # Delete the extracted files directory
    shutil.rmtree(directory_to_extract_to)
    return sheets    


########### get_sheet_name_ends ##############
 '''   


##########column_rename_starts##########
def column_rename(df={},*args):
    '''
    Renames the columns inside the dataframe
    
    Parameter : dictionary
    
    Return:dictonary 
      
    '''
    
    for x,y in df.items():
        if('healthy' in x):
            y.columns =['bandwidth','day1','day2','day3','day4','day5','day6','day7','day8','day9','day10']
        else:
            y.columns =['bandwidth','day1','day2','day3','day4','day5','day6','day7','day8','day9']
            
    #print('done')
    return df
#########columns_rename_ends################



##########column_category_creattion_starts########
def col_cat_creator(df={},*args):
    
    '''
    creating column and category for dataframes merging together
    
    Parameter : dictionary
    
    Return:dictonary 
        
    '''
    for x,y in df.items():
        new = re.split(r'_',x)
        y['class'] = new[0]
        y['category'] = new[1]
        
    return df
##########column_category_creattion_ends######## 
  
##########combine_all_starts###########
def combine_all(df={},*args):
    '''
    Combines all the dataframes together
    
    '''
    
    final_df = pd.concat(df.values(),ignore_index=True,sort=False)
    
    ##exporting the dataframne for checking    
    #export_excel = final_df.to_excel(r"C:\Users\BharatAgri\Birac\export_dataframe.xlsx",index = None, header =True)
    #print('file created')
    #final_df.to_excel(index=False)
    return(final_df)
##########combine_all_ends###########   
 
    
######missing_value_starts######
def missing_value(df):
    
    '''
    Treating the missing value
    
    Parameter: dataframe
    
    Returns: dataframe
    
    
    '''
    res = df.isnull().sum()
    #print(type(res))
    for x,v in res.iteritems():
        if (v!=0):
            df[x].fillna(0, inplace = True)
        else:
            continue
    
    return df
    #print(df.isnull().sum())

#####missing_value_ends########   
    
    
''' 
###ref_normalizer_starts###
def ref_normalizer(df):
    
    
    Normalizes the reflectance value
    
    Parameter: dataframe
    
    Returns: dataframe
    

    
    column_names = df.columns
    for x in column_names:
        print(df[x].shape)
    #for x, v in column_names.iteritems():
         #print(df.x)
             
    # mod_ref = df.apply(lambda x: x/95 if x.name in col else x)
    #return mod_ref
    
###ref_normalizer_ends####
 '''  
    
    
    
############## preprocessing function starts #################      
def preprocessing():
    
    """ preprocessing for reflectance data 
    
    parameters : none,
    
    returns: preprocessed dataframes healthy and inoculated  
    
    
    
    """
    
     ###reading data 
    print('preprocessing block starts')
    
    ##declaration
    input_paths = list()
    input_paths = config_parse()
    #dataFrame = list()
    healthy = list()
    inoculated= list()
    data = {}
    #final_df = pd.DataFrame()
    
    
    for d in input_paths:
        #df.append(pd.read_excel(d))
        xls = xlrd.open_workbook(d, on_demand=True)
    
    sheet_name = xls.sheet_names()
    
    #creating the dataframe list 
    for s in sheet_name:
        healthy.append(s+'_healthy')
        inoculated.append(s+'_inoculated')
        
    
    
    for d in input_paths:
        for s in sheet_name:
            if(d=='RH.xls'):              
                data[s+'_healthy'] = pd.read_excel(d,sheet_name=s)
                #dataFrame.append(pd.read_excel(d,sheet_name=s))
            else:
                data[s+'_inoculated'] = pd.read_excel(d,sheet_name=s)
                #dataFrame.append(pd.read_excel(d,sheet_name=s))
                #print('else')
    
    
    ##columns renaming
    data = column_rename(data)
    #for x,y in data.items():
       # print(x)
       
    ##column and category creation 
    data = col_cat_creator(data)
    
    ##combining all the dataframes together
    final_df = combine_all(data)
    
    ##treating_missing_values
    final_df = missing_value(final_df)
    
    ##normalizer for reflectance data 
     
    #final_df = ref_normalizer(final_df)
    
    #exporting for check
    #final_df.to_excel(r"C:\Users\BharatAgri\Birac\export_dataframe.xlsx",index = None, header =True)
    print(final_df.shape)
    
    
          
   
    print('preprocessing block ends')
    
################ preprocessing function ends #################
   
    
    
    
    
    
    
######## main function Starts ############
def main():
    
    print("data science ")
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

    
    
    
    
    
    
    
    