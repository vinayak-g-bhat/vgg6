# vgg6
VGG6 model experiments on CIFAR-10 database


# Prerequisites:
Python 3.12+ 
wandb api key for charting

# installation
pip install requirements

# configuration
change the hyper parameters in main.py
NOTE: change download to True for the first run in main.py:
 train_data = datasets.CIFAR10('./data', train=True, transform=trans_t, download=False)
test_data = datasets.CIFAR10('./data', train=False, transform=trans, download=False)

device = torch.device('mps' if torch.cuda.is_available() else 'cpu') ==> change to the device code of your machine.
macbook m2 = mps

Optimizer block needs to be configured separately in main by commenting and uncommenting

#


