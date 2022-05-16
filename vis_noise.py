"""
script for visualizing mnist & cifar with noise added
"""
from main import get_split
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
import torch
import os

noise_lvls = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
# noise type is randomly set pixels to 2 * st. dev. px value wrt whole image
# i.e. a value corresponding to near white / full intensity
two_sd = 0

for i, noise_lvl in enumerate(noise_lvls):
    # _, test_data = get_split("mnist", noise_type=two_sd, noise_lvl=noise_lvl)
    
    # less noise_lvl needed for cifar10 to be a similar classification difficulty for human a>
    _, test_data = get_split("cifar10", noise_type=two_sd, noise_lvl=noise_lvl/3)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
                             
    for _, (img, target) in enumerate(test_loader):
        save_image(img, f"mnist_{i}.png")
        if _ == 1: break
