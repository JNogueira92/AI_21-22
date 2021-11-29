import matplotlib.pyplot as plt
import numpy as np
import os

DATASET = 'dataset_test'
DATASET_PATH = 'dataset'
RESULTS_PATH = 'results'
MODEL = 'model.pt'

NUMBER_INPUTS = 6
NUMBER_HUMAN_INPUTS = 3

MARIO_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['left'],
    ['left', 'A'],
    ['A'],
]

HUMAN_INPUTS = [
    [0, 0, 0], #No Action
    [0, 0, 1], #right
    [0, 1, 1], #right jump
    [1, 0, 0], #left
    [1, 1, 0], #left jump
    [0, 1, 0], #jump
]

INPUTS = [
    [1, 0, 0, 0, 0, 0], #No Action
    [0, 1, 0, 0, 0, 0], #right
    [0, 0, 1, 0, 0, 0], #right jump
    [0, 0, 0, 1, 0, 0], #left
    [0, 0, 0, 0, 1, 0], #left jump
    [0, 0, 0, 0, 0, 1], #jump
]



def plot_loss(epochs,y_loss_training, y_loss_testing):
    # y -> Loss
    # x -> Epochs
    i = 1
    x_epochs = []

    while i <= epochs:
        x_epochs.append(i)
        i += 1

    x = np.array(x_epochs)

    # --- TRAINING LINE ----
    # plotting line 1 points
    plt.plot(x, y_loss_training, label="training")

    # --- TESTING LINE ----
    # plotting line 2 points
    plt.plot(x, y_loss_testing, label="testing")

    # x axis label
    plt.xlabel('epoch')
    # y axis label
    plt.ylabel('loss')
    # Set the title of the current axes.
    plt.title('Loss')
    # show a legend on the plot
    plt.legend()
    # SAVE IMAGE
    file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Results/Loss.png")
    plt.savefig(file_name) # -----> Location and name of the image
    # clear the plot
    plt.clf()



def plot_acc(epochs,y_accuracy_training, y_accuracy_testing):
    # y -> Accuracy
    # x -> Epochs
    i = 1
    x_epochs = []

    while i <= epochs:
        x_epochs.append(i)
        i += 1

    x = np.array(x_epochs)

    # --- TRAINING LINE ---- ----
    # plotting line 1 points
    line2 = plt.plot(x, y_accuracy_training, label="training")

    # --- TESTING LINE ----
    # plotting line 2 points
    line1 = plt.plot(x, y_accuracy_testing, label="testing")
    # x axis label
    plt.xlabel('epoch')
    # y axis label
    plt.ylabel('accuracy (%)')
    # Set a title for the axes
    plt.title('Accuracy')
    # show a legend on the plot
    plt.legend()
    # SAVE IMAGE
    file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Results/Acc.png")
    plt.savefig(file_name)  # -----> Location and name of the image
    # clear the plot
    plt.clf()
