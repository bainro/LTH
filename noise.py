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

# need to modulo the gpu # from done_i
for trial in range(n_repeats):
    exp_name = "trial_" + str(trial)
    cmd = "python main.py --dataset cifar10 --arch_type lenet5 --end_iter 35 --last_iter_epochs 35 --batch_size 200 --exp_name "
    cmd += exp_name
    q_l.append(['/bin/bash', '-c', cmd])

done_i = 0
for i, process in enumerate(q_l):
    print("done_i: ", done_i)
    # could break & be unefficient on other PCs
    gpu_i = done_i % 3
#     if done_i < 4:
#         gpu_i = 2
#     elif done_i < 6:
#         gpu_i = 1
#     else:
#         gpu_i = 0
    process[-1] += (" --gpu " + str(gpu_i) + " &>/tmp/" + str(i))
    # for testing!!!
    # process[-1] = "echo LOL"
    p = Popen(process)
    processes.append((i%N, p))
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
