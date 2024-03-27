# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 21:43:33 2022

@author: SYEDIHTESHAMHUSSAINS
"""
'''
In this code we have done 
(1) Resizing (224,224) (see function in line 84)
(2) saved image in a single format (.jpg) (line 86 and 99 )
(3) Augumentation: it is done to increase the dataset (see the last for loop)
'''

from PIL import Image
import cv2 as cv
import os
import imgaug.augmenters as iia

Input_path = r'C:\Users\SYEDIHTESHAMHUSSAINS\Desktop\data_set\Original_Dataset'


#Original_Dataset folder contatins following directories 
#  1.Reception, 2.waiting area, 3.Injection Room, 4.Danger Room, 5.diagnostic


path = r'C:\Users\SYEDIHTESHAMHUSSAINS\Desktop\data_set\Augumented_Dataset'



#  Path represent the output directory where we want to save our new data

#  1st we creat the same folder as the input path contains 

agumentation= iia.Sequential([ 
    
    #Flip
    iia.Fliplr(0.5),
    
    #Affine
    
    iia.Affine(translate_percent={"x":(-0.2 , 0.2), "y":(-0.2 , 0.2)},
               
               rotate= (-30 , 30),
               scale=(0.5 , 1.5)  ),
    
    #Mulitply
    
    
    iia.Multiply((0.8 , 1.2)),
    
    #Linear Contrast
    
    iia.LinearContrast((0.6 , 1.4)),
    
    
    #Perform blow methods sometime (not all the time)
    
    iia.Sometimes(0.5,
                  
                  #Gaussain Blur
                  iia.GaussianBlur((0.0 , 3.0))  
                  )
    
     
    ])
    


for dir in os.listdir(Input_path): 
    
    c=1 # c reprents the images numeric value i.e img1 , img2, etc

    inside=str(Input_path)+'/' + str(dir)
    #  "str(dir)" provide a specific path information inside of Input_path  
    #   Direcroty i.e "1.Reception" ,  "2.waiting area" etc
    
    Output_path= str(path)+'/'+str(dir) 
    if not os.path.exists(Output_path):
        os.mkdir(Output_path)         
    # It creat similar directories (as Input_path contains) to the output path    
             
    for filename in os.listdir(inside):
        pth=inside+'/'+str(filename) # it provide real address (folder address + file name)
        img = Image.open(pth)
        img=img.convert('RGB')
        img=img.resize((224,224), Image.ANTIALIAS)
        
        name= 'Aug'+str(c)+'.jpg'
        img.save(os.path.join(Output_path,name))
        c+=1
        #above cmd save original image into the distination folder
                
        ''' resizeng work with ".Image function" while augumentation is using 
        ".cv" function , that is why we opend the same image throuh cv.imread 
        that we have saved through img.save cmd above. '''
        
        Img=cv.imread(os.path.join(Output_path, name))
        
        for _ in range (4): 
            agumented_image = agumentation(image= Img)
            name= 'Aug'+str(c)+'.jpg'      #because c has new value after saving image "img" above      
            cv.imwrite(os.path.join(Output_path, name), agumented_image)        
            c+=1
            continue