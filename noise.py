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
n_repeats = 10
q_l = []

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
                
# avg of 10 for sparsity vs acc with RLT + WLT on MNIST + CIFAR10
# load each checkpoint & do noisy testing
    # multiple types of noise
    # sparsity vs noise score for RLT + WLT on MNIST + CIFAR10
# train single WLT & check for overfitting
    # will require spliting training into ""+ validation set
# doc in non-official repo about bp diff & freezing potentially affecting non-masked elements (?)
