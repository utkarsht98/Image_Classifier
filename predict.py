# PROGRAMMER: Utkarsh .T.
# DATE CREATED: 29-12-2018
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: To predict the name of the flower(based on probability)
#          given the path to the flower to the file "predict.py".
# Use argparse Expected Call with <> indicating expected user input:
#      python predict.py <path to the image> --gpu
#   Example call:
#    python predict.py flowers/test/10/image_03010.jpg --arch vgg16 --gpu

import argparse
import numpy as np
import torch
from torch import nn
from torch import optim
import torchvision
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchvision import datasets, transforms, models
from PIL import Image
import time
import json

def main():
    input_args = get_input_args()
    with open(input_args.category_flower_names, 'r') as f:
        cat_to_name = json.load(f)
    image_location = input_args.dir
    checkpoint_location = input_args.load_dir + '/checkpoint.pth'
    gpu = input_args.gpu
    device = 'cpu'
    if gpu and torch.cuda.is_available():
        device = 'cuda'
    print("\n-----------Prediction Argument Information-------------\n")
    print("Path to image: ",input_args.dir)
    print("Checkpoint location: ",input_args.load_dir)
    print("GPU enabled: ",input_args.gpu)
    print("Is Cuda available: ",torch.cuda.is_available())
    print("Device going to be used: ",device.upper())
    print("TOPK value: ",input_args.top_k)
    print("Category File name: ",input_args.category_flower_names)
    print("\n---------------------------------------------------\n")

    # loading checkpoint.....
    model = load_checkpoint(checkpoint_location)

    print("\n-------------Performing Prediction--------------------\n")
    prob, classes, flower_name = predict(image_location, model, cat_to_name, input_args.top_k)

    print("Probability:\t{}".format(prob))
    print("Classes:\t\t{}".format(classes))
    print("Names(top {}):\t\t{}".format(input_args.top_k,flower_name))

def get_input_args():
    parser =  argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, help='location of image to test')
    parser.add_argument('--load_dir',default='checkpoints', help='checkpoint directory')
    parser.add_argument('--top_k', type=int, default=5, help='K most likely classes')
    parser.add_argument('--category_flower_names', type=str, default='cat_to_name.json', help='Mapping of categories to real name')
    parser.add_argument('--gpu', action='store_true', help='use GPU')

    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("Wrong choice of architecture\n")
    for param in model.parameters():
        param.requires_grad = False
    learn_rate= checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    """classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 10000)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.1)),
                          ('fc2', nn.Linear(10000, 4096)),
                          ('relu', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.1)),
                          ('fc3', nn.Linear(4096, 1024 )),
                          ('relu', nn.ReLU()),
                          ('fc5', nn.Linear(1024, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))"""
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    image_process = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])

    np_image = image_process(img)
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model, cat_to_name, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    # TODO: Implement the code to predict the class from an image file
    image.unsqueeze_(0)
    outputs = model(image)
    probs = torch.exp(outputs)
    top_probs, top_labs = probs.topk(topk, dim=1)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]

    return top_probs, top_labs, top_flowers

if __name__ == '__main__':
    main()
