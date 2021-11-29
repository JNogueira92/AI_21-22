import numpy as np
import time
import os
import pickle
import gzip
import sys
import random
import time
import torch
import torchvision
from torch import nn
from  utils import DATASET_PATH, DATASET, RESULTS_PATH, MODEL, HUMAN_INPUTS, INPUTS, MARIO_MOVEMENT, NUMBER_INPUTS, NUMBER_HUMAN_INPUTS, \
        plot_acc,plot_loss

DEBUG = True

BATCH_SIZE = 32
EPOCHS = 100
TRAIN_SPLIT = 0.75

def joinDatasets(data1, data2):
    data1.extend(data2)
    return data1

def loadDataset(path = DATASET_PATH, file = DATASET):
    if os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)),path, file)):
        with gzip.open(os.path.join(os.path.dirname(os.path.realpath(__file__)),path, file), 'rb') as f:
            data = pickle.load(f)
            if DEBUG:
                print("DATASET LOADED FROM: ", os.path.join(os.path.dirname(os.path.realpath(__file__)),path, file))
    else:
        data = list()
    
        
    return data

def balanceDataset(data):

    if DEBUG:
        print("BALANCING DATASET")

    new_data = data.copy()

    actions = []
    actions = [[] for item in range(NUMBER_INPUTS)]

    
    for i in range(NUMBER_INPUTS):
        actions[i] = [index for index, element in enumerate(new_data) if (np.array(element[1]) == np.array(INPUTS[i])).all()]
    
    #get most recurrent action
    max_actions = max([len(element) for _, element in enumerate(actions)])
    

    if DEBUG:
        print("OLD DATASET COMPOSITION: ")
        for i in range(NUMBER_INPUTS):
            print(MARIO_MOVEMENT[i], ": ", len(actions[i]))

    #remove no action
    new_data = [element for index, element in enumerate(new_data) if index not in actions[0]]
    
    if DEBUG:
        print("DATASET BALANCE REMOVED NOOP:", len(actions[0]))

    #remove deaths and bad actions
    deaths = [index for index, element in enumerate(new_data) if element[3] < -1 ]
    new_data = [element for index, element in enumerate(new_data) if index not in deaths]

    if DEBUG:
        print("DATASET BALANCE REMOVED DEATHS:", len(deaths))

    #update indexes
    for i in range(NUMBER_INPUTS):
        actions[i] = [index for index, element in enumerate(new_data) if (np.array(element[1]) == np.array(INPUTS[i])).all()]

    '''
    #decrease common event right by half
    decrease_number = round(len(actions[1])/2)
    to_delete = set(random.sample(range(len(actions[1])), decrease_number))
    actions[1] = [x for i,x in enumerate(actions[1]) if i not in to_delete]
    new_data = [element for index, element in enumerate(new_data) if index not in actions[1]]
    if DEBUG:
        print("DATASET BALANCE REMOVED COMMONS:", len(to_delete))
    '''

    #multiply rare events
    for d in data:
        #right jump
        if np.array_equal(d[1], INPUTS[2]):
            new_data += (d,) * 1#(round(max_actions/len(actions[2]))-1)
        #left
        elif np.array_equal(d[1], INPUTS[3]):
            new_data += (d,) * 4
        #left jump
        elif np.array_equal(d[1], INPUTS[4]):
            new_data += (d,) * 4
        #jump
        elif np.array_equal(d[1], INPUTS[5]):
            new_data += (d,) * 5


    if DEBUG:

        actions_counter = np.zeros(NUMBER_INPUTS, np.int32)

        for d in new_data:
            if (np.array(d[1]) == np.array(INPUTS[0])).all() :
                actions_counter[0]+=1
            elif (np.array(d[1]) == np.array(INPUTS[1])).all() :
                actions_counter[1]+=1
            elif (np.array(d[1]) == np.array(INPUTS[2])).all() :
                actions_counter[2]+=1
            elif (np.array(d[1]) == np.array(INPUTS[3])).all() :
                actions_counter[3]+=1
            elif (np.array(d[1]) == np.array(INPUTS[4])).all() :
                actions_counter[4]+=1
            elif (np.array(d[1]) == np.array(INPUTS[5])).all() :
                actions_counter[5]+=1

        print("NEW DATASET COMPOSITION: ")
        for i in range(NUMBER_INPUTS):
            print(MARIO_MOVEMENT[i], ": ", actions_counter[i])

    
    return new_data

class CustomTensorDataset(torch.utils.data.TensorDataset):

            def __init__(self, x, y):
                super().__init__(x,y)
                self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.Grayscale(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize(84),
                    torchvision.transforms.Normalize(0, 255)
                ])

            def __getitem__(self, index):
                #return self.transform(self.tensors[0][index]), self.tensors[1][index]
                return (self.transform(self.tensors[0][index]),) + tuple(t[index] for t in self.tensors[1:])

"""
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, NUMBER_INPUTS),
            """


class NeuralNetwork:
    def __init__(self,datasets = [],epochs = EPOCHS, batch_size = BATCH_SIZE, train_split = TRAIN_SPLIT, restore=False):
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            torch.nn.Conv2d(1, 32, 8, 4),
            torch.nn.BatchNorm2d(32),
            torch.nn.ELU(),
            torch.nn.Dropout2d(0.5),
            torch.nn.Conv2d(32, 64, 4, 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ELU(),
            torch.nn.Dropout2d(0.5),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ELU(),
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(64 * 7 * 7),
            torch.nn.Dropout(),
            torch.nn.Linear(64 * 7 * 7, 120),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(120),
            torch.nn.Dropout(),
            torch.nn.Linear(120, NUMBER_INPUTS),
        )
    
        if restore:
            model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),RESULTS_PATH, MODEL)
            self.model.load_state_dict(torch.load(model_path))

        self.datasets = datasets
        self.model.eval()
        self.model = self.model.to(self.dev)
        self.epochs = epochs
        self.batch = batch_size
        self.split = train_split
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        #variables for plotting
        self.y_loss_training = []
        self.y_loss_testing = []
        self.y_acc_training = []
        self.y_acc_testing = []

    def createDataset(self):
        if not self.datasets:
            data = loadDataset()
        else:
            data = []
            for ds in self.datasets:
                new_dataset = loadDataset(ds[0],ds[1])
                data = joinDatasets(data,new_dataset)

        if not data:
            sys.exit("No data found!")

        data = balanceDataset(data)
            
        random.shuffle(data)

        states, actions, _, _, _ = map(np.array, zip(*data))

        # permute [H, W, C] array to [C, H, W] tensor
        states = np.moveaxis(states, 3, 1)

        # train dataset
        input_train = states[:int(len(states) * self.split)]
        output_train = actions[:int(len(actions) * self.split)]

            

        train_set = CustomTensorDataset(torch.tensor(input_train), torch.tensor(output_train,dtype=torch.float))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch,
                                                shuffle=True, num_workers=2)

        # test dataset
        input_val, output_val = states[int(len(input_train)):], actions[int(len(output_train)):]

        val_set = CustomTensorDataset(torch.tensor(input_val), torch.tensor(output_val,dtype=torch.float))

        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch,
                                                shuffle=False, num_workers=2)

        return train_loader, val_loader

    def train(self):

        train_loader, val_order = self.createDataset()
        
        start_time = time.time()

        for epoch in range(self.epochs):
            if DEBUG:
                print("TRAINING EPOCH: ", epoch+1, " of ", self.epochs)

            self.train_epoch(train_loader)

            self.test(val_order)

            self.saveModel()

        end_time = time.time()

        if DEBUG:
            print("TRAIN ENDED TIME ELAPSED: ", (end_time - start_time))
        # ---------> CREATING THE PLOTS <----------
        # Display and Save the results
        #LOSS
        loss_training = np.array(self.y_loss_training)
        loss_testing = np.array(self.y_loss_testing)
        plot_loss(self.epochs,loss_training, loss_testing)
        #ACCURACY
        acc_training = np.array(self.y_acc_training)
        acc_testing = np.array(self.y_acc_testing)
        plot_acc(self.epochs,acc_training, acc_testing)


    def train_epoch(self,data_loader):
        self.model.train()  # set model to training mode
        current_loss, current_acc = 0.0, 0.0

        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(self.dev), labels.to(self.dev)  # send to device

            self.optimizer.zero_grad()  # zero the parameter gradients

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)  # forward
                _, predictions = torch.max(outputs, 1)
                loss = self.loss_function(outputs, labels)

                loss.backward()  # backward
                self.optimizer.step()

            current_loss += loss.item() * inputs.size(0)  # statistics
            _,targets = torch.max(labels.data,1)
            current_acc += torch.sum(predictions == targets)

        total_loss = current_loss / len(data_loader.dataset)
        total_acc = current_acc / len(data_loader.dataset)

        if DEBUG:
            print("TRAINING Loss: ",total_loss ," Accuracy: ", total_acc.item())
        
        # Appending values to the arrays (for the PLOT)
        self.y_loss_training.append(total_loss)
        self.y_acc_training.append(total_acc * 100)


    def test(self, data_loader):

        self.model.eval()  # set model in evaluation mode

        current_loss, current_acc = 0.0, 0.0

        # iterate over the validation data
        for i, (inputs, labels) in enumerate(data_loader):
            # send the input/labels to the GPU
            inputs, labels = inputs.to(self.dev), labels.to(self.dev)

            # forward
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, predictions = torch.max(outputs, 1)
                loss = self.loss_function(outputs, labels)

            # statistics
            current_loss += loss.item() * inputs.size(0)
            _,targets = torch.max(labels.data,1)
            current_acc += torch.sum(predictions == targets)

        total_loss = current_loss / len(data_loader.dataset)
        total_acc = current_acc / len(data_loader.dataset)

        if DEBUG:
                print("TESTING Loss: ",total_loss ," Accuracy: ", total_acc.item())
            
        # Appending values to the arrays (for the PLOT)
        self.y_loss_testing.append(total_loss)
        self.y_acc_testing.append(total_acc.item() * 100)

    def saveModel(self):
        # save model
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),RESULTS_PATH, MODEL)
        torch.save(self.model.state_dict(), model_path)
        if DEBUG:
            print("MODEL SAVED: ", model_path)

if __name__ == '__main__':
    #datasets = [("dataset-1","level_1_game1"),("dataset-1","level_1_game2"),("dataset-1","level_1_game3"),("dataset-1","level_1_game4"),("dataset-1","level_1_game5"),("dataset-1","level_1_game6"),("dataset-1","level_1_game7")]
    datasets = [("dataset","dataset_test_old")]
    neuralNet = NeuralNetwork(datasets=datasets)
    #neuralNet = NeuralNetwork()
    neuralNet.train()