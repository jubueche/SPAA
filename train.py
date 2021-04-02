import argparse
import importlib
from datajuicer import run, configure, make_unique
import os.path
import os

ap = argparse.ArgumentParser()
ap.add_argument("-exp", nargs="+", default=[])
ap.add_argument("-n_threads", type=int, default=1)
ap.add_argument("-force", action='store_true')

flags, settings =ap.parse_known_args()
settings = {key[1:]:value for key, value in [arg.split("=") for arg in settings]}

toplevel = importlib.import_module("Experiments")

if flags.exp==[]:
    for f in os.listdir("Experiments"):
        if os.path.isfile(os.path.join("Experiments",f)) and f.endswith(".py"):
            flags.exp+=[f[0:-3]]

for module in ["Experiments."+ex for ex in flags.exp]:
    importlib.import_module(module)

experiments = [getattr(getattr(toplevel,ex),ex) for ex in flags.exp]

models = []

for ex in experiments:
    models += ex.train_grid()

models = configure(make_unique(models),settings)
if flags.force:
    run_mode = "force"
else:
    run_mode = "normal"
run(models, "train", n_threads=flags.n_threads, run_mode=run_mode)("{*}")