"""
script for testing the noise robustness of ~Lenet5 WLTs & RLTs on MNIST & CIFAR10 
"""
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import numpy as np
import time
import os

def check_for_done(l):
    for i, (_i, p) in enumerate(l):
        if p.poll() is not None:
            return True, i
    return False, False

processes = list()
N = 6
n_repeats = 10
q_l = []

'''
for trial in range(n_repeats):
    exp_name = "trial_" + str(trial)
    cmd = "python main.py --dataset cifar10 --arch_type fc1 --end_iter 35 --last_iter_epochs 35 --batch_size 200 --exp_name "
    cmd += "wlt_" + exp_name
    q_l.append(['/bin/bash', '-c', cmd])
    cmd = "python main.py --dataset mnist --arch_type fc1 --end_iter 35 --last_iter_epochs 35 --batch_size 200 --exp_name "
    cmd += "wlt_" + exp_name
    q_l.append(['/bin/bash', '-c', cmd])
    cmd = "python main.py --dataset cifar10 --arch_type fc1 --end_iter 35 --last_iter_epochs 35 --batch_size 200 --rlt --exp_name "
    cmd += "rlt_" + exp_name
    q_l.append(['/bin/bash', '-c', cmd])
    cmd = "python main.py --dataset mnist --arch_type fc1 --end_iter 35 --last_iter_epochs 35 --batch_size 200 --rlt --exp_name "
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
                print("done_i: ", done_i)
                print("done_p: ", done_p)
                wait = False
            else:
                # set so the cpu can chill
                time.sleep(0.5)
                
# avg of 10 tickets. Sparsity vs acc with WLT + RLT on MNIST + CIFAR10
DPI = 1200
prune_iterations = 35

dump_dir = f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}/"
for arch_type in tqdm(arch_types):
    # loads 1 of 40 trials... Need to avg over 10 similar for this dataset 
    _RENAME = os.path.join(dump_dir, "rlt_" + exp_name, "lt_bestaccuracy.dat")
    d = np.load(_RENAME, allow_pickle=True)
    b = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}/lt_bestaccuracy.dat", allow_pickle=True)
    c = np.load(f"{os.getcwd()}/dumps/lt/{arch_type}/{dataset}/reinit_bestaccuracy.dat", allow_pickle=True)

    a = np.arange(prune_iterations)
    plt.plot(a, b, c="blue", label="Winning tickets") 
    plt.plot(a, c, c="red", label="Random tickets") 
    plt.title(f"Test Accuracy vs Weights % ({arch_type} | {dataset})") 
    plt.xlabel("Weights %") 
    plt.ylabel("Test accuracy") 
    plt.xticks(a, d, rotation ="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray") 

    plt.savefig(f"{os.getcwd()}/plots/lt/combined_plots/combined_{arch_type}_{dataset}.png", dpi=DPI, bbox_inches='tight') 
    plt.close()
        
# load each checkpoint & do noisy testing
    # multiple types of noise
    # sparsity vs noise score for RLT + WLT on MNIST + CIFAR10
# train single WLT & check for overfitting
    # will require spliting training into ""+ validation set
