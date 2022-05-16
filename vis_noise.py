"""
script for visualizing mnist & cifar with noise added
"""
from main import get_split, get_model, test
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

datasets = ["mnist", "cifar10"]
noise_lvls = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
# noise type is randomly set pixels to 2 * st. dev. px value wrt whole image
two_sd = 0

for noise_lvl in noise_lvls:
    print("noise_lvl: ", noise_lvl)
    _traindata, testdata = get_split("mnist", noise_type=two_sd, noise_lvl=noise_lvl, log>
    # less noise_lvl needed for cifar10 to be a similar classification difficulty for human a>
    #_traindata, testdata = get_split("cifar10", noise_type=noise_type, noise_lvl=noise_lvl/3)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=512, shuffle=False, num_wo>
    trial_sum += test(model, test_loader)

