# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:21:00 2020

@author: BharatAgri
"""

# Pythono3 code to rename multiple  
# files in a directory or folder 
  
# importing os module 
import os 
  
# Function to rename multiple files 
def main(): 
    i = 0
      
    for filename in os.listdir('D:\\Project BIG\\classifiers\\ri_leaf\\xyz\\'):
        dst ="Hostel" + str(i) + ".bmp"
        src =filename 
        dst ='xyz'+ dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 