# Lottery Ticket Hypothesis in Pytorch 

This repository contains a **Pytorch** implementation of the paper [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) by [Jonathan Frankle](https://github.com/jfrankle) and [Michael Carbin](https://people.csail.mit.edu/mcarbin/) that can be **easily adapted to any model/dataset**.
		
## Requirements
```
pip3 install -r requirements.txt
```
## How to run the code ? 
### Using datasets/architectures included with this repository :
```
python3 main.py --prune_type=lt --arch_type=fc1 --dataset=mnist --prune_percent=10 --prune_iterations=35
```
- `--prune_type` : Type of pruning  
	- Options : `lt` - Lottery Ticket Hypothesis, `reinit` - Random reinitialization
	- Default : `lt`
- `--arch_type`	 : Type of architecture
	- Options : `fc1` - Simple fully connected network, `lenet5` - LeNet5, `AlexNet` - AlexNet, `resnet18` - Resnet18, `vgg16` - VGG16 
	- Default : `fc1`
- `--dataset`	: Choice of dataset 
	- Options : `mnist`, `fashionmnist`, `cifar10`, `cifar100` 
	- Default : `mnist`
- `--prune_percent`	: Percentage of weight to be pruned after each cycle. 
	- Default : `10`
- `--prune_iterations`	: Number of cycle of pruning that should be done. 
	- Default : `35`
- `--lr`	: Learning rate 
	- Default : `1.2e-3`
- `--batch_size`	: Batch size 
	- Default : `60`
- `--end_iter`	: Number of Epochs 
	- Default : `100`
- `--print_freq`	: Frequency for printing accuracy and loss 
	- Default : `1`
- `--valid_freq`	: Frequency for Validation 
	- Default : `1`
- `--gpu`	: Decide Which GPU the program should use 
	- Default : `0`

## Interesting papers that are related to Lottery Ticket Hypothesis which I enjoyed 
- [Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask](https://eng.uber.com/deconstructing-lottery-tickets/)

