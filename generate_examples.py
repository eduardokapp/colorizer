# This module implements several functions to deal with images and transform them to LAB
import torch
from skimage import io, color
import matplotlib.pyplot as plt
import pickle
import numpy as np

# we'll use unpickle to load from the cifar-10 dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        some_dict = pickle.load(fo, encoding='bytes')
    return some_dict

# misc function to read any image and return its rgb and lab versions, as tensors
def read_and_convert(img_path):
    rgb_img =  io.imread(fname=img_path)
    lab_img = color.rgb2lab(rgb_img/255).astype('float32')
    return(torch.from_numpy(rgb_img), torch.from_numpy(lab_img))

# misc function that plots the original image along its components
# either RGB or LAB
def plot_channels(img, kind='rgb'):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(20,10))
    ax0.imshow(img)
    ax1.imshow(extract_color(img, 0, kind))
    ax2.imshow(extract_color(img, 1, kind))
    ax3.imshow(extract_color(img, 2, kind))

# Extract one of the three channels from either RGB or LAB color spaces
def extract_color(img, where=0, kind='rgb'):
    new_img = img.clone().detach()
    zero_out = [x for x in range(0, img.shape[2]) if x != where]
    new_img[:,:,zero_out] = 0
    if kind == 'lab':
        if where!=0:
            new_img[:,:,0] = 70
        new_img = torch.from_numpy(color.lab2rgb(np.asarray(new_img)))
    return new_img



# reading images
for idx in range(1,5+1):
    batch_data = unpickle(f'./data/cifar-10-batches-py/data_batch_{idx}')
    if idx == 1:
        data = batch_data[b'data']
    else:
        data = np.vstack((data, batch_data[b'data']))
batch_data = None

# splitting X and Y
data = torch.from_numpy(data)
data = data.reshape(data.shape[0],3,32,32)
data = np.array(torch.permute(data, [0,2,3,1]))

# convert all images to lab color space
# TODO: maybe optimize? LGTM for now
lab_data = data.copy()
for idx in range(0, lab_data.shape[0]):
    lab_data[idx,:,:,:] = color.rgb2lab(lab_data[idx,:,:,:])

# now we split them between L and A,B
# recall that L is in the first channel
train_x = lab_data[:,:,:,0].copy()
train_y = lab_data[:,:,:,[1,2]].copy()

# save
torch.save(torch.from_numpy(train_x), './train/train_x.pt')
torch.save(torch.from_numpy(train_y), './train/train_y.pt')

# for the test examples, we simply operate on the test batch sample for cifar-10
# I'll do the exact same operations as before
test_batch = unpickle('./data/cifar-10-batches-py/test_batch')[b'data']
test_batch = test_batch.reshape(test_batch.shape[0], 3, 32, 32)
test_batch = test_batch.transpose([0,2,3,1])
lab_test = test_batch.copy()
for idx in range(0, lab_test.shape[0]):
    lab_test[idx,:,:,:] = color.rgb2lab(lab_test[idx,:,:,:])

test_x = lab_test[:,:,:,0].copy()
test_y = lab_test[:,:,:,[1,2]].copy()

# save
torch.save(torch.from_numpy(test_x), './test/test_x.pt')
torch.save(torch.from_numpy(test_y), './test/test_y.pt')
