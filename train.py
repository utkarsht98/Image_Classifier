# PROGRAMMER: Utkarsh .T.
# DATE CREATED: 27-12-2018
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: To train a network based on the given training
#          dataset of flowers and to print the training loss,
#          validation loss and validation accuracy.Also a checkpoint
#          has to be created to save the network for "predict.py".
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py --dir <directory with images> --arch <model> --gpu
#   Example call:
#    python train.py --dir flowers --arch vgg16 --gpu
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import os
from collections import OrderedDict
import sys
import time
import json

def main():
    loss_set = []
    accuracy_set = []
    print("-----------Training started----------")
    input_args = get_input_args()
    if input_args.arch = 'vgg16':
        input_size = 25088
    else:
        input_size = 1024
    data_directory = input_args.dir
    save_dir = input_args.save_dir
    gpu = input_args.gpu
    device = 'cpu'

    if gpu and torch.cuda.is_available():
        device = 'cuda'

    print("\n----------Architecture Information-------------\n")
    print("Image source dir name: ",input_args.dir)
    print("Checkpoint location dir name: ",save_dir)
    print("Model Architecture being used: ",input_args.arch)
    print("GPU enabled: ",input_args.gpu)
    print("Is Cuda available: ",torch.cuda.is_available())
    print("Device going to be used: ",device.upper())
    print("EPOCHS value: ",input_args.epochs)
    print("Hidden Unit[layer1,layer2,layer3]: ",input_args.hidden_units)
    print("Learn Rate: ",input_args.learning_rate)
    print("\n------------------------------------------------\n")

    if not os.path.exists(save_dir):
        print("Creating directory for checkpoint.")
        os.makedirs(save_dir)

        #loading data into three sets- train,test and valid datasets
    train_data, validation_data, test_data, train_dataloaders, valid_dataloaders, test_dataloaders = load_data(input_args.dir)

    model, classifier = create_network(input_args.arch, input_args.hidden_units)#creating the Classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = input_args.learning_rate)#Defining the optimizer and method for error calculation

    # Training the model
    if device == 'cuda':
        print("\nTraining using CUDA as our device...\n")
        loss_set,accuracy_set = train_network('cuda', model, input_args.epochs, criterion, optimizer, train_dataloaders,valid_dataloaders)
    else:
        print("\nTraining using CPU as our device...\n")
        loss_set, accuracy_set = train_network('cpu', model, input_args.epochs, criterion, optimizer, train_dataloaders, valid_dataloaders)

    correct = 0
    total = 0
    model.to('cuda')
    with torch.no_grad():
        for images,labels in test_dataloaders:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images).to('cuda')
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    print("\n-----------Model Accuracy-------------\n")
    print("Validation Loss: {:.4f}\nAccuracy: {:.4f}\nAccuracy(in %): {:.2f}%".format(loss_set[-1], accuracy_set[-1], accuracy_set[-1] * 100))

    print("\n---------Creating Checkpoint-----------\n")
    save_checkpoint(input_args.arch,input_size,input_args.epochs, model, optimizer, classifier, criterion, input_args.learning_rate, train_data, save_dir)

def get_input_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--dir',help='location of input source for images')
    parser.add_argument('--arch', type=str, default='vgg16', help='model architecture to be used: <vgg16>')
    parser.add_argument('--epochs', type=int, default='15', help='number of epochs to use')
    parser.add_argument('--hidden_units', nargs='+', type=int, default=[10000, 4096, 1024], help='hidden layers - only 3(for densenet between 1024 and 102)')
    parser.add_argument('--learning_rate', type=float, default='0.00001', help='learning rate for the network')
    parser.add_argument('--save_dir', help='loaction to save checkpoint', default='checkpoints')
    parser.add_argument('--gpu', action='store_true', help='use GPU')
    return parser.parse_args()

def load_data(image_dir_name):
    # Loading the data
    data_dir = image_dir_name
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_data_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=test_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=64,shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=32, shuffle=False)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=32, shuffle=False)

    return train_image_datasets, valid_image_datasets, test_image_datasets, train_dataloaders, valid_dataloaders, test_dataloaders

def create_network(arch_name, hidden_layers):
    if arch_name = 'vgg16':
        input_size = 25088
    else:
        input_size = 1024
    output_size = 102
    model = None
    if(arch_name == 'vgg16'):
        model = models.vgg16(pretrained=True)
    elif(arch_name == 'densenet121'):
        model = models.densenet121(pretrained=True)
    else:
        print("Provided architecture must be either densenet121 or vgg16")
        return None

    for param in model.parameters():
        param.require_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_layers[0])),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.1)),
                          ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
                          ('relu', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.1)),
                          ('fc3', nn.Linear(hidden_layers[1], hidden_layers[2] )),
                          ('relu', nn.ReLU()),
                          ('fc5', nn.Linear(hidden_layers[2], output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    print("------------Neural Network Model Created-------------")
    model.classifier = classifier
    return model, classifier

def validation(device, model, valid_dataloaders, criterion):
    model.to(device)
    test_loss = 0
    accuracy = 0
    for images, labels in valid_dataloaders:

        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.cuda.FloatTensor).mean()

    return test_loss, accuracy

def train_network(device, model, epochs, criterion, optimizer, train_dataloaders, valid_dataloaders):
    print("The device to be used is: {}\n".format(device.upper()))
    print("Training started...\n")

    loss_set = []
    accuracy_set =[]

    start_time = time.time()

    print_every = 40
    steps = 0

    model.to(device)

    for e in range(epochs):
        running_loss = 0
        model.train()
        for ii, (images, labels) in enumerate(train_dataloaders):
            steps += 1

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(device, model, valid_dataloaders, criterion)
                loss_set.append(test_loss/len(valid_dataloaders))
                accuracy_set.append(accuracy/len(valid_dataloaders))
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(valid_dataloaders)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(valid_dataloaders)))

                running_loss = 0
                model.train()
    end_time = time.time()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:", tot_time ,"in seconds")
    hours = int( (tot_time / 3600) )
    minutes = int( ( (tot_time % 3600) / 60 ) )
    seconds = int( ( (tot_time % 3600) % 60 ) )
    print("\n** Total Elapsed Runtime:", str(hours) + ":" + str(minutes) + ":" + str(seconds))

    return loss_set, accuracy_set

def save_checkpoint(arch, input_size, epochs, model, optimizer,classifier,criterion,learning_rate,train_image_datasets,save_directory):
    checkpoint = {
              'input_size': input_size,
              'output_size': 102,
              'arch': arch,
              'learning_rate': learning_rate,
              'state_dict': model.state_dict(),
              'class_idx': train_image_datasets.class_to_idx,
              'optimizer': optimizer.state_dict(),
              'epochs': epochs,
              'classifier': classifier
             }
    torch.save(checkpoint, save_directory+'/checkpoint.pth')
    #call to main function to run the program
if __name__ == "__main__":
    main()
