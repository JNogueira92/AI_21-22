from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
import numpy as np
import time
import os
import pickle
import gzip
import torchvision
import torch
from train import NeuralNetwork
from  utils import DATASET_PATH, DATASET, HUMAN_INPUTS, INPUTS, MARIO_MOVEMENT, NUMBER_INPUTS, NUMBER_HUMAN_INPUTS

import matplotlib.pyplot as plt

DEBUG = True

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

if __name__ == '__main__':
    global close
    close = False
    done = False
    restart = False
    total_reward = 0
    agent = NeuralNetwork(restore = True)

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, MARIO_MOVEMENT)
    
    env.render()
    env.unwrapped.viewer._window.on_key_press = keyPress
    state = env.reset()
    while(not close and not done):
        if restart:
            restart = False
            env.reset()
            total_reward = 0
            if DEBUG:
                print("RESTART")

        transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.Grayscale(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize(84),
                    torchvision.transforms.Normalize(0, 255)
                ])
        
        state = np.moveaxis(state, 2, 0)  # channel first image
        state = torch.from_numpy(np.flip(state, axis=0).copy())  # np to tensor
        state = transform(state).unsqueeze(0)  # apply transformations
        
        #if DEBUG:
            #image = state.squeeze(1)
            #plt.imshow(  image.permute(1, 2, 0)  )

        state = state.to(agent.dev)  # add additional dimension

        with torch.set_grad_enabled(False):  # forward
            outputs = agent.model(state)

        normalized = torch.nn.functional.softmax(outputs, dim=1)
        action = np.argmax(normalized.cpu().numpy()[0])

        if DEBUG:
            print("AGENT ACTION: ", MARIO_MOVEMENT[action])

        state, reward, done, info = env.step(action)

        total_reward += reward
        env.render()
        time.sleep(0.005)

        