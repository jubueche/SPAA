# SPAA
Repository containing the code for Spiking Probabilistic Adversarial Attacks.

## Setup
- Install cleverhans using ```pip install git+https://github.com/cleverhans-lab/cleverhans.git#egg=cleverhans```
- Install Sinabs ```pip install sinabs```
- Install pytorch ```pip install torch ujson gdown```
- Install aermanager ```pip install aermanager```
- Install pygraphviz ```sudo apt-get install python-dev graphviz libgraphviz-dev pkg-config``` followed by ```pip install pygraphviz```

## Tutorial
Start by executing ```python tutorial_NMNIST.py``` to see if everything works fine.

## Setting up a new experiment
See the ```Experiments/example_experiment.py```. Use ```quick_access.py``` to call the experiment when you want to debug for example.
Also see the ```@cachable``` function in ```experiment_utils.py``` for an example of how to implement functions for experiments.