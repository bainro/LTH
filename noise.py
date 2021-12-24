"""
script for testing the noise robustness of ~Lenet5 WLTs & RLTs on MNIST & CIFAR10
"""
from subprocess import Popen, PIPE
import time

def check_for_done(l):
    for i, p in enumerate(l):
        if p.poll() is not None:
            return True, i
    return False, False

processes = list()
N = 5

# stdout=PIPE, stderr=PIPE
q_l = [['/bin/bash', 'sleep', '5'], ['/bin/bash', 'sleep', '5'], ['/bin/bash', 'sleep', '5'], ['/bin/bash', 'sleep', '5'], ['/bin/bash', 'sleep', '5'], ['/bin/bash', 'echo', 'OhMehLOL!']]
queue = list(q_l)
# ["echo lol", "sleep 5", "echo dblol", "sleep 5", "sleep 5", "sleep 5", "sleep 5", "echo trplol", "sleep 5", "echo duadlol"]
for process in queue:
    p = Popen(process)
    processes.append(p)
    if len(processes) == N:
        wait = True
        while wait:
            done, num = check_for_done(processes)

            if done:
                # need to check gpu on process cmd call
                finished_p = processes.pop(num)
                print("finished_p: ", finished_p)
                wait = False
            else:
                # set so the cpu can save cycles checking
                time.sleep(0.5)
