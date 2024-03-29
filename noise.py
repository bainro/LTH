"""
script for testing the noise robustness of FC1 WLTs & RLTs on MNIST & CIFAR10 
"""
from main import get_split, get_model, test
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import numpy as np
import torch
import time
import os

def check_for_done(l):
    for i, (_i, p) in enumerate(l):
        if p.poll() is not None:
            return True, i
    return False, False

DPI = 1200
prune_iterations = 35
datasets = ["mnist", "cifar10"]
processes = list()
N = 6 # number of || processes
n_repeats = 7
q_l = []

'''
for trial in range(n_repeats):
    exp_name = "trial_" + str(trial)
    #cmd = "python main.py --dataset cifar10 --arch_type fc1 --end_iter 35 --last_iter_epochs 35 --batch_size 200 --exp_name "
    #cmd += "wlt_" + exp_name
    #q_l.append(['/bin/bash', '-c', cmd])
    cmd = "python main.py --dataset mnist --arch_type lenet5 --end_iter 35 --last_iter_epochs 35 --batch_size 200 --exp_name "
    cmd += "wlt_" + exp_name
    q_l.append(['/bin/bash', '-c', cmd])
    #cmd = "python main.py --dataset cifar10 --arch_type fc1 --end_iter 35 --last_iter_epochs 35 --batch_size 200 --rlt --exp_name "
    #cmd += "rlt_" + exp_name
    #q_l.append(['/bin/bash', '-c', cmd])
    cmd = "python main.py --dataset mnist --arch_type lenet5 --end_iter 35 --last_iter_epochs 35 --batch_size 200 --rlt --exp_name "
    cmd += "rlt_" + exp_name
    q_l.append(['/bin/bash', '-c', cmd])
'''

# keeps 2 training threads per gpu
done_i = None
for i, process in enumerate(q_l):
    
    # will break &| be unefficient on other PCs
    if done_i == None:
        gpu_i = i % 3
        print("i: ", i)
    else:
        gpu_i = done_i % 3
        print("done_i: ", done_i)

    process[-1] += (" --gpu " + str(gpu_i) + " &>/tmp/" + str(i))

    p = Popen(process)
    processes.append((gpu_i, p))
    if len(processes) == N:
        wait = True
        while wait:
            done, num = check_for_done(processes)

            if done:
                # need to check gpu on process cmd call
                done_i, done_p = processes.pop(num)
                wait = False
            else:
                # set so the cpu can chill
                time.sleep(0.5)
        
# avg of n_repeat (eg 8) tickets. Sparsity vs acc with WLT + RLT on MNIST + CIFAR10
for dataset in []:
    dump_dir = f"{os.getcwd()}/dumps/lt/lenet5/{dataset}/"
    
    a = np.arange(prune_iterations)
    
    for n in range(n_repeats):
        _b = os.path.join(dump_dir, "rlt_trial_" + str(n), "lt_bestaccuracy.dat")
        b = np.load(_b, allow_pickle=True)
        if n == 0:
            total_b = b
        else:
            for i in range(len(b)):
                total_b[i] += b[i]
        # sanity check:
        # print("total_b: ", total_b)
    avg_b = total_b / n_repeats

    for n in range(n_repeats):
        _c = os.path.join(dump_dir, "wlt_trial_" + str(n), "lt_bestaccuracy.dat")
        c = np.load(_c, allow_pickle=True)
        if n == 0:
            total_c = c
        else:
            for i in range(len(c)):
                total_c[i] += c[i]
    avg_c = total_c / n_repeats
    
    d = np.load(os.path.join(dump_dir, "rlt_trial_0", "lt_compression.dat"), allow_pickle=True)
    
    y_min = min(np.concatenate((avg_b, avg_c)))
    y_max = max(np.concatenate((avg_b, avg_c)))
    
    plt.plot(a, avg_b, c="blue", label="Random tickets") 
    plt.plot(a, avg_c, c="red", label="Winning tickets") 
    plt.title(f"Test Accuracy vs Weights % (LeNet5 | {dataset})") 
    plt.xlabel("Weights %") 
    plt.ylabel("Test accuracy") 
    plt.xticks(a, d, rotation="vertical") 
    plt.ylim(y_min - 1, y_max + 1)
    plt.legend() 
    plt.grid(color="gray") 

    plt.savefig(f"{os.getcwd()}/plots/lt/lenet5/{dataset}/avg_over_{n_repeats}_trials.png", dpi=DPI, bbox_inches='tight') 
    plt.close()
        
# load each checkpoint & do noisy testing
for noise_type in [0]: #range(4):
    print("===============")
    print("noise_type: ", noise_type)
    for noise_lvl in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        print("noise_lvl: ", noise_lvl)
        trial_sum = 0
        for trial in range(n_repeats):
            logdir = f"{os.getcwd()}/plots/mnist_lenet5_{noise_type}_{noise_lvl}_{trial}/"
            exists = os.path.exists(logdir)
            if not exists:
                  os.makedirs(logdir)
            _traindata, testdata = get_split("mnist", noise_type=noise_type, noise_lvl=noise_lvl, logdir=logdir)
            # less noise_lvl needed for cifar10 to be a similar classification difficulty for human annotators (i.e. me!).
            #_traindata, testdata = get_split("cifar10", noise_type=noise_type, noise_lvl=noise_lvl/3)
            test_loader = torch.utils.data.DataLoader(testdata, batch_size=512, shuffle=False, num_workers=2, drop_last=False)

            # 100% dense
            # model = torch.load(f"/home/rbain/git/LTH2/saves/lenet5/mnist/wlt_trial_{trial}/0_model_lt.pth.tar")
            
            # WLT ~80% pruned
            # model = torch.load(f"/home/rbain/git/LTH2/saves/lenet5/mnist/wlt_trial_{trial}/12_model_lt.pth.tar")
            # RLT ~80% pruned
            # model = torch.load(f"/home/rbain/git/LTH2/saves/lenet5/mnist/rlt_trial_{trial}/12_model_lt.pth.tar")

            # WLT ~70% pruned
            # model = torch.load(f"/home/rbain/git/LTH2/saves/lenet5/mnist/wlt_trial_{trial}/9_model_lt.pth.tar")
            # RLT ~70% pruned
            # model = torch.load(f"/home/rbain/git/LTH2/saves/lenet5/mnist/rlt_trial_{trial}/9_model_lt.pth.tar")
            
            # WLT 96.7% pruned
            model = torch.load(f"/home/rbain/git/LTH2/saves/lenet5/mnist/wlt_trial_{trial}/26_model_lt.pth.tar")
            # RLT 96.7% pruned
            # model = torch.load(f"/home/rbain/git/LTH2/saves/lenet5/mnist/rlt_trial_{trial}/26_model_lt.pth.tar")
            
            trial_sum += test(model, test_loader)
            
        avg_acc = trial_sum / n_repeats
        print("avg trial acc: ", avg_acc)
