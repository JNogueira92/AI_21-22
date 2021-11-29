from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
import numpy as np
import time
import os
import sys
import pickle
import gzip
from  utils import DATASET_PATH, DATASET, HUMAN_INPUTS, INPUTS, MARIO_MOVEMENT, NUMBER_INPUTS, NUMBER_HUMAN_INPUTS

DEBUG = True

def loadDataset():
    if os.path.exists(os.path.join(DATASET_PATH, DATASET)):
        with gzip.open(os.path.join(DATASET_PATH, DATASET), 'rb') as f:
            data = pickle.load(f)
    else:
        data = list()

    return data

if __name__ == '__main__':

    done = False
    total_reward = 0
    
    data = loadDataset()

    if not data:
        sys.exit("Data not found!")

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, MARIO_MOVEMENT)
    
    env.render()

    state = env.reset()

    for action in data:
        state, reward, done, info = env.step(action[1].index(1))
        
        total_reward += reward
        
        if DEBUG:
            print("ACTION: ", action[1].index(1), " REWARD: ", reward, " TOTAL REWARD: ", total_reward)

        env.render()
        time.sleep(0.005)        

    env.close()
    if DEBUG:
        print("FINISHED TOTAL REWARD: ", total_reward)