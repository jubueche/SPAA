# SPAA
Repository containing the code for Spiking Probabilistic Adversarial Attacks.

## Setup
- Install cleverhans from https://github.com/jubueche/cleverhans using ```git clone https://github.com/jubueche/cleverhans```, ```cd cleverhans``` and ```pip install -e .```
- Install Sinabs ```pip install sinabs```
- Install pytorch ```pip install torch```

## Setting up a new experiment
See the ```Experiments/example_experiment.py```. Use ```quick_access.py``` to call the experiment when you want to debug for example.
Also see the ```@cachable``` function in ```experiment_utils.py``` for an example of how to implement functions for experiments.