"""
script for testing the noise robustness of FC1 WLTs & RLTs on MNIST & CIFAR10 
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

DPI = 1200
prune_iterations = 35
datasets = ["mnist", "cifar10"]
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
                
# avg of n_repeat (eg 10) tickets. Sparsity vs acc with WLT + RLT on MNIST + CIFAR10
for dataset in tqdm(datasets):
    dump_dir = f"{os.getcwd()}/dumps/lt/fc1/{dataset}/"
    
    a = np.arange(prune_iterations)
    
    for n in range(n_repeats):
        _b = os.path.join(dump_dir, "rlt_trial_" + str(n), "lt_bestaccuracy.dat")
        b = np.load(_b, allow_pickle=True)
        if n == 0:
            total_b = b
        else:
            for i in len(b):
                total_b[i] += b[i]
    avg_b = total_b / n_repeats

    for n in range(n_repeats):
        _c = os.path.join(dump_dir, "wlt_trial_" + str(n), "lt_bestaccuracy.dat")
        c = np.load(_c, allow_pickle=True)
        if n == 0:
            total_c = c
        else:
            for i in len(c):
                total_c[i] += c[i]
    avg_c = total_c / n_repeats
    
    d = np.load(os.path.join(dump_dir, "rlt_trial_0", "lt_compression.dat"))

    plt.plot(a, avg_b, c="blue", label="Random tickets") 
    plt.plot(a, avg_c, c="red", label="Winning tickets") 
    plt.title(f"Test Accuracy vs Weights FC | % ({dataset})") 
    plt.xlabel("Weights %") 
    plt.ylabel("Test accuracy") 
    plt.xticks(a, d, rotation="vertical") 
    plt.ylim(0,100)
    plt.legend() 
    plt.grid(color="gray") 

    plt.savefig(f"{os.getcwd()}/plots/lt/combined_plots/avg_fc1_{dataset}.png", dpi=DPI, bbox_inches='tight') 
    plt.close()
        
# load each checkpoint & do noisy testing
    # multiple types of noise
    # sparsity vs noise score for RLT + WLT on MNIST + CIFAR10
# (separate script) train single WLT & check for overfitting
    # will require spliting training into ""+ validation set
