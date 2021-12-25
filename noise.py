"""
script for testing the noise robustness of ~Lenet5 WLTs & RLTs on MNIST & CIFAR10
"""
from subprocess import Popen, PIPE
import time

def check_for_done(l):
    for i, (_i, p) in enumerate(l):
        if p.poll() is not None:
            return True, i
    return False, False

processes = list()
N = 6
n_repeats = 2 # 10
q_l = []

for trial in range(n_repeats):
    exp_name = "trial_" + str(trial)
    cmd = "python main.py --dataset cifar10 --arch_type lenet5 --end_iter 35 --last_iter_epochs 35 --batch_size 200 --exp_name "
    cmd += exp_name
    q_l.append(['/bin/bash', '-c', cmd])
    cmd = "python main.py --dataset mnist --arch_type lenet5 --end_iter 35 --last_iter_epochs 35 --batch_size 200 --exp_name "
    cmd += exp_name
    q_l.append(['/bin/bash', '-c', cmd])
    cmd = "python random_pruning.py --dataset cifar --arch_type lenet5 --end_iter 35 --last_iter_epochs 35 --batch_size 200 --exp_name "
    cmd += exp_name
    q_l.append(['/bin/bash', '-c', cmd])
    cmd = "python random_pruning.py --dataset mnist --arch_type lenet5 --end_iter 35 --last_iter_epochs 35 --batch_size 200 --exp_name "
    cmd += exp_name
    q_l.append(['/bin/bash', '-c', cmd])

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
                # done_i = done_i % N
                print("done_i: ", done_i)
                print("done_p: ", done_p)
                wait = False
            else:
                # set so the cpu can save cycles checking
                time.sleep(0.5)
