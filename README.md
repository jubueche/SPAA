# SPAA
Repository containing the code for Spiking Probabilistic Adversarial Attacks.

## Setup
- Install cleverhans using ```pip install git+https://github.com/cleverhans-lab/cleverhans.git#egg=cleverhans```
- Install Sinabs ```pip install sinabs```
- Install pytorch ```pip install torch ujson gdown matplotlib tonic```
- Install aermanager ```pip install aermanager```
- Install sinabs-dynapcnn ```pip install sinabs-dynapcnn==0.2.1.dev53```

## Tutorial
Start by executing ```python tutorial_NMNIST.py``` to see if everything works fine.

## Setting up a new experiment
See the ```Experiments/example_experiment.py```. Use ```quick_access.py``` to call the experiment when you want to debug for example.
Also see the ```@cachable``` function in ```experiment_utils.py``` for an example of how to implement functions for experiments.

## How to run dynapcnn_test.py experiment on the chip
1. Install the latest samna: `pip install samna --index-url https://gitlab.com/api/v4/projects/27423070/packages/pypi/simple -U`
2. Install sinabs on the master branch. Typically `pip install sinabs` should do the trick here, otherwise pull the repo manually and install.
3. Install the hardware backend repository for dynapcnndevkit/speck2b `sinabs-dynapcnn` on the `add_reset_method` branch. If that branch does not exist, it has probably been merged into master.
4. In the dynapcnn_test.py script, change the variable DYNAPCNN_HARDWARE to your hardware model, e.g. dynapcnndevkit or speck2b
