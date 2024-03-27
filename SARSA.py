 # -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:34:24 2022

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
        

def learn(State, State2, reward, Action, Action2):
    
    state= list_states.index(State)
    state2= list_states.index(State2)
    action= list_actions.index(Action)
    action2= list_actions.index(Action2)
    
    predict = Q[state, action]
    target= reward + gamma * Q[state2, action2]
    Q[state, action]= Q[state, action] + lr_rate * (target - predict)



total_episodes = 10000
max_steps = 20
reward_table = reward_function()
rewards=0

for episode in range(total_episodes):
    
    t=0
    state=start_state
    
    action= choose_action(state)
    
    #print('new_spisode')
    for t in range (max_steps ):
        
        state2, reward, done = step(state, action)
        
        
        action2 = choose_action(state2)
        
        learn(state, state2, reward, action, action2 )
        
        #print('state=', state, 'action=', action, 'state2=', state2, 'action2=', action2 )
        state= state2 
        action =  action2 
        
        t+=1 
        rewards+=1
        #if state2 == 'sT':
        #   break
        
        


    
#print ("Score over time: ", rewards/total_episodes)
#print(Q)


 
    
 
messages = {'a01': "You are in Acceptance Room, Please move to waiting Room",
            'a02': "You are in Acceptance Room, Please move to Hot waiting Room",
            'a13': "You are in waiting Room, Please move to Hot Injection Room",
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
            'a11safe': "you are in Hot waiting room, Please move to the diagnostic Room",
            'a1211': "you are still in Hot waiting room, Please move to the diagnostic Room",
            'a12T': "System dosen't work", }
    
print('.....................................................................................')
print('State', '  ', 'Action', '     ','Massege') 
print('.....................................................................................')
for i in range(n_states):
    action_index = np.argmax(Q[i, :]) #
    action=list_actions[action_index]
    print(list_states[i],'     ', action, '       ',messages[action]  )
