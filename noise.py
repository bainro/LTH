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
N = 3

q_l = [['/bin/bash', '-c', 'sleep 5'], ['/bin/bash', '-c', 'sleep 5'], ['/bin/bash', '-c', 'sleep 5'], ['/bin/bash', '-c', 'echo OMG']]
queue = list(q_l)
for i, process in enumerate(queue):
    p = Popen(process)
    processes.append((i, p))
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
                # set so the cpu can save cycles checking
                time.sleep(0.5)
