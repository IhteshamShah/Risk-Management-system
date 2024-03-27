# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:25:38 2022

@author: SYEDIHTESHAMHUSSAINS
"""


"""

First hold the image in front of camera and then run the program. 
Because!
At the first run it automatically capture the image from the camera 

"""


import warnings
import cv2
import numpy as np 
from keras.models import load_model
warnings.filterwarnings("ignore")

model = load_model('resnet_model.h5')

import SARSA
  
    

def Pridiction_of_state(current_Room, previous_state):  # Pridiction Of State
    
    if current_Room == 'Acceptance_R': 
        state='s0'
        
    if current_Room == 'waiting_R': 
        if previous_state == 's0':
            state = 's1'
        elif previous_state == 's1':
            state = 's4'
        elif previous_state == 's10':
            state = 's1'
        elif previous_state == 's3':
            state = 's7'
        elif previous_state == 's7':
            state = 's8'
            
    if current_Room == 'Injection_R': 
        if previous_state == 's1':
            state = 's3'
        elif previous_state == 's4':
            state = 's3'
    
    if current_Room == 'Hot_waiting_R': 
        if previous_state == 's3':
            state = 's6'
        elif previous_state == 's7':
            state = 's6'
        elif previous_state == 's8':
            state = 's6'
        elif previous_state == 's6':
            state = 's12'
        elif previous_state == 's2':
            state = 's10'
    
    if current_Room == 'Diagnostic_R': 
        if previous_state == 's6':
            state = 's11'
        elif previous_state == 's12':
            state = 's11'
            
    return state


messages = {'a01': "You are in Acceptance Room, Please move to waiting Room",
            'a02': "You are in Acceptance Room, Please move to Hot waiting Room",
            'a13': "You are in waiting Room, Please move to Injection Room",
            'a14': "You are moving in the same waiting Room, Please move to Injection Room", 
            'a210': "You are moving to hot wating room without being injected, BE CAREFUL ! ",
            'a36': "You are in the Injection Room, Please move to Hot waiting Room",
            'a37': "You are moving back to the Waiting Room, Please move to Hot waiting Room",
            'a38': "You are moving back to the Waiting Room, Please move to Hot waiting Room",
            'a43': "You are in waiting Room, Please move to Injection Room",
            'a4T': "System dosen't work",
            'a611': "You are in Hot Waiting Room, Please move to the Diagnostic Room",
            'a612': "You are still in Hot Waiting Room, Please move to the Diagnostic Room",
            'a76': "Your are Injected and in wating Room, please move to the Hot waiting Room",
            'a78': "Your are Injected and in wating Room, please move to the Hot waiting Room",
            'a86': "Your are Injected and in wating Room, please move to the Hot waiting Room",
            'a8T': "System dosen't work",
            'a101': "You are in Hot wating room without injected, please move to waiting room for your turn",
            'a10T': "System dosen't work",
            'a11safe': "Congrats Process has been completed",
            'a1211': "you are still in Hot waiting room, Please move to the diagnostic Room",
            'a12T': "System dosen't work", }
    



frameWidth = 224  #Height and width of the test picture for prediction
frameHeight = 224 
classnames = ['Acceptance_R','waiting_R', 'Injection_R', 'Hot_waiting_R','Diagnostic_R'] #classes (Rooms) on with the DL model is trained




previous_state = '0' # Before patient is accepted

print('.....................................................................................')
print('State', '  ', 'Action', '     ','Massege') 
print('.....................................................................................')


while True :    
 
    cap = cv2.VideoCapture(0) #Capture video from laptop cam (0)represents the basic cam
    success,img = cap.read()  # if the video is successfuly captured then take a frame and store it in img 
    img = cv2.resize(img, (frameWidth, frameHeight)) # resize the image (224x224) was the size on which we traied the resnet50 model
    img=np.expand_dims(img,axis=0) # shape the image for pridiction

    predict_room=model.predict(img) # it allocate the binary values against each class
    output_class=classnames[np.argmax(predict_room)] # Heighest binary value represnts the predicted class
    
    Current_Room = str(output_class) # predicted class represnts the current room
    Current_state = Pridiction_of_state(Current_Room, previous_state) #state prediction in each room   
    
    State_index= SARSA.list_states.index(Current_state) # index of the current state becuase we have state as string form e.g 's0', 's2' etc.
    Optimal_action_index = np.argmax(SARSA.Q[State_index, :]) # optimal action index
    Optimal_action=SARSA.list_actions[Optimal_action_index] # Index of action is again converted into string form of action
    previous_state = Current_state
    
    
    
    print(Current_state,'     ', Optimal_action, '       ',messages[Optimal_action]  )
    
    if Optimal_action == 'a11safe' :
        break_while= input('Do you wanna Continue for another Patient Y/N   ')
        if break_while == 'N':
            break
    else:
        input('Be ready for another capture')

        
