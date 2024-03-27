# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:06:26 2022

@author: SYEDIHTESHAMHUSSAINS
"""

import numpy as np
import random
import time, pickle, os

epsilon = 0.3
# min_epsilon = 0.1
# max_epsilon = 1.0
# decay_rate = 0.01


lr_rate = 0.81
gamma = 0.96

list_actions=['a01','a02','a13','a14', 'a210', 'a36', 'a37', 'a38', 'a43', 'a4T', 'a611', 'a612', 'a76', 'a78', 'a86', 'a8T', 'a101', 'a10T', 'a11safe', 'a1211', 'a12T']

list_states= ['s0', 's1', 's2', 's3', 's4', 's6', 's7', 's8', 'sT', 's10', 's11', 's12', 'ssafe']

start_state= 's0'
n_actions=int(len(list_actions))
n_states=int(len(list_states))

Q = np.zeros((n_states, n_actions))           


def reward_function():
    '''
    Reward for being in state state_int.

    state_int: State integer. int.
    -> Reward.
    '''
    reward_table= np.zeros((len(list_states),len(list_actions), len(list_states)))
    
    
    reward_table[list_states.index('s11'), 
                 list_actions.index('a11safe'), 
                 list_states.index('ssafe')] = 10     #reward for last safe state=10
    
    reward_table[: , : , list_states.index('sT')] = -10     #reward for last terminal state= -10
    
    return reward_table


def choose_action(state):
    if (np.random.uniform(0, 1) < epsilon):
          if state == 's0':
              action= random.choice(['a01','a02'])
          elif state == 's1':
              action= random.choice(['a13','a14'])
          elif state == 's2':
              action= 'a210'
          elif state == 's3':
              action= random.choice(['a36','a37','a38'])
          elif state == 's4':
              action= random.choice(['a43','a4T'])
          elif state == 's6':
              action= random.choice(['a611','a612'])
          elif state == 's7':
              action= random.choice(['a76','a78'])
          elif state == 's8':
              action= random.choice(['a86','a8T'])
          elif state == 's10':
              action= random.choice(['a101','a10T'])
          elif state == 's11':
              action= 'a11safe'
          elif state == 's12':
              action= random.choice(['a12T','a1211'])
          else:
              action=random.choice(list_actions)

    else:
        action_index = np.argmax(Q[list_states.index(state), :])
        action= list_actions[action_index]
    return action

def step(state, action):
    ''' 
    last charactors in the actions tell us the next state 
    i.e a611 tells next state would be 11,
    action contains multiple charactors i.e a10T, asafe a1112 etc. 
    information of the snext state depends on the length of chartactor values 
    '''
    done= False
    charactor_in_action=list(action)    # convert action in to list e.g 'a611' to ['a', '6', '1', '1']
    if len(charactor_in_action)==3 :    #length of the list
        state2='s'+charactor_in_action[2]  # return state2 as s3,s4,s7 etc
    elif len(charactor_in_action) > 3 :    #for those states where actions are more then 3 charactors
        if int(charactor_in_action[1]) > 5: # In all the actions where 2nd entry > 5  
            conv=''.join(charactor_in_action[2:]) #next state would be all the intries > 2nd index
            state2='s'+conv
        else:
            conv=''.join(charactor_in_action[3:]) #next state would be all the intries > 2nd index
            state2='s'+conv
            
            
    reward= reward_table[list_states.index(state),list_actions.index(action),list_states.index(state2)] 
    if state2 == 'sT' or 'ssafe':
        done=True
           
    return state2,reward, done

reward_table = reward_function()