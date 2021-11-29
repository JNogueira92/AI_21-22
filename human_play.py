from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
import numpy as np
import time
import os
import pickle
import gzip
from  utils import DATASET_PATH, DATASET, HUMAN_INPUTS, INPUTS, MARIO_MOVEMENT, NUMBER_INPUTS, NUMBER_HUMAN_INPUTS

DEBUG = False

def loadDataset():
    if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)),DATASET_PATH, DATASET)):
        with gzip.open(os.path.join(os.path.dirname(os.path.realpath(__file__)),DATASET_PATH, DATASET), 'rb') as f:
            data = pickle.load(f)
    else:
        data = list()

    return data

def saveDataset(data):
    data_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),DATASET_PATH, DATASET)
    print("Saving dataset to " + data_file_path)

    if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)),DATASET_PATH)):
        os.mkdir(os.path.join(os.path.dirname(os.path.realpath(__file__)),DATASET_PATH))

    with gzip.open(data_file_path, 'wb') as f:
        pickle.dump(data, f)

    return 0

def keyPress(key,mod):
    global restart, close, human_action
    
    if DEBUG:
        print("KEY PRESSED: ", key)

    if key == 65307:
        #ESC exit
        close = True
    elif key == 114:
        #r restart
        restart = True
    elif key==97:
        #left
        human_action[0] = 1
    elif key == 100:
        #right
        human_action[2] = 1
    elif key == 111:
        #A (jump)
        human_action[1] = 1


def keyRelease(key,mod):
    global human_action

    if DEBUG:
        print("KEY RELEASED: ", key)

    if key==97:
        #left
        human_action[0] = 0
    elif key == 100:
        #right
        human_action[2] = 0
    elif key == 111:
        #A (jump)
        human_action[1] = 0

    

def mapHumanAction(input):
    
    if DEBUG:
        print("MAPPING ACTION: ", input)

    if (input == HUMAN_INPUTS[1]).all():
        #right
        return INPUTS[1]
    elif (input == HUMAN_INPUTS[2]).all():
        #right jump
        return INPUTS[2]
    elif (input == HUMAN_INPUTS[3]).all():
        #left
        return INPUTS[3]
    elif (input == HUMAN_INPUTS[4]).all():
        #left jump
        return INPUTS[4]
    elif (input == HUMAN_INPUTS[5]).all():
        #jump
        return INPUTS[5]
    return INPUTS[0]
    

if __name__ == '__main__':
    global human_action,close,restart
    close = False
    restart = False
    done = False
    total_reward = 0
    
    data = loadDataset()

    human_action = np.zeros(NUMBER_HUMAN_INPUTS,dtype=np.int32)

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, MARIO_MOVEMENT)
    
    env.render()
    env.unwrapped.viewer._window.on_key_press = keyPress
    env.unwrapped.viewer._window.on_key_release = keyRelease

    state = env.reset()
    while(not close and not done):
        if restart:

            if DEBUG:
                print("RESTART")

            state = env.reset()
            restart = False
            data = loadDataset()
        
        action = np.copy(human_action)
        old_state = state
    
        state, reward, done, info = env.step(mapHumanAction(human_action).index(1))
        
        if DEBUG:
            print("ACTION: ", mapHumanAction(action).index(1), " REWARD: ", reward)

        data.append((old_state,mapHumanAction(action),state,reward,done))
        
        total_reward += reward
        
        env.render()
        time.sleep(0.005)

    env.close()
    print(total_reward)
    #saveDataset(data)