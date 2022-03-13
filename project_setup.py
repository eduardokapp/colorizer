# -*- coding: utf-8 -*-
"""
Download dataset and make dirs

@author: Eduardo Kapp
"""

import torchvision
import os

# make necessary dirs
os.makedirs("./data", exist_ok=True)
os.makedirs("./train", exist_ok=True)
os.makedirs("./test", exist_ok=True)

# one liner
torchvision.datasets.CIFAR10(root='./data', download=True)

print("Done!")