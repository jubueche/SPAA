import argparse
import importlib
import os.path
import os

ap = argparse.ArgumentParser()
ap.add_argument("-exp", nargs="+", default=[])

flags =ap.parse_args()

toplevel = importlib.import_module("Experiments")

if flags.exp==[]:
    for f in os.listdir("Experiments"):
        if os.path.isfile(os.path.join("Experiments",f)) and f.endswith(".py"):
            flags.exp+=[f[0:-3]]

for module in ["Experiments."+ex for ex in flags.exp]:
    importlib.import_module(module)

experiments = [getattr(getattr(toplevel,ex),ex) for ex in flags.exp]

for ex in experiments:
    ex.visualize()