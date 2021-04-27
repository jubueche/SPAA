import os
import os.path
from datajuicer import cachable, get, format_template
import argparse
import random
from networks import load_ann, get_prob_net, get_mnist_ann_arch, get_prob_net_continuous, get_summed_network, load_gestures_snn


def standard_defaults():
    return {}


def help():
    return {"No arguments": "No arguments"}


launch_settings = {
    "direct": "python {code_file} {args}",
    "bsub": 'mkdir -p Resources/Logs; bsub -o Resources/Logs/{session_id}.log -W 1:00 -n 8 -R "rusage[mem=1024]" "python3 {code_file} {args}"',
}


def mk_runner(architecture, env_vars):
    @cachable(
        dependencies=[
            "model:" + key for key in architecture.default_hyperparameters().keys()
        ],
        saver=None,
        loader=architecture.loader,
        checker=architecture.checker,
        table_name=architecture.__name__,
    )
    def runner(model):
        try:
            mode = get(model, "mode")
        except:
            mode = "direct"
        model["mode"] = mode

        def _format(key, value):
            if type(value) is bool:
                if value:
                    return f"-{key}"
                else:
                    return ""
            else:
                return f"-{key}={value}"

        model["args"] = " ".join(
            [
                _format(key, get(model, key))
                for key in list(architecture.default_hyperparameters().keys())
                + env_vars
                + ["session_id"]
            ]
        )
        command = format_template(model, launch_settings[mode])
        print(command)
        os.system(command)
        return None

    return runner


def _get_flags(default_dict, help_dict):
    parser = argparse.ArgumentParser()
    for key, value in default_dict.items():
        if type(value) is bool:
            parser.add_argument(
                "-" + key, action="store_true", help=help_dict.get(key, "")
            )
        else:
            parser.add_argument(
                "-" + key, type=type(value), default=value, help=help_dict.get(key, "")
            )
    parser.add_argument("-session_id", type=int, default=0)

    flags = parser.parse_args()
    if flags.session_id == 0:
        flags.session_id = random.randint(1000000000, 9999999999)

    return flags

class IBMGestures:
    @staticmethod
    def make():
        d = IBMGestures.default_hyperparameters()

        def mk_data_dir(mode="direct"):
            if mode == "direct":
                return "data/Gestures/"
            elif mode == "bsub":
                return "$SCRATCH/data/Gestures/"
            raise Exception("Invalid Mode")

        d["mk_data_dir"] = mk_data_dir
        d["data_dir"] = "{mk_data_dir({mode})}"
        d["code_file"] = "main_IBMGestures.py"
        d["architecture"] = "IBMGestures"
        d["train"] = mk_runner(IBMGestures, ["data_dir"])
        return d

    @staticmethod
    def default_hyperparameters():
        d = standard_defaults()
        return d

    @staticmethod
    def checker(sid, table, cache_dir):
        return True

    @staticmethod
    def get_flags():
        default_dict = {
            **IBMGestures.default_hyperparameters(),
            **{"data_dir": "data/Gestures/"},
        }
        return _get_flags(default_dict, help())

    @staticmethod
    def loader(sid, table, cache_dir):
        data = {}
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, f"Resources/Models/{sid}_model.pt")
        snn = load_gestures_snn(model_path)
        data["ann"] = None
        data["snn"] = snn
        data["prob_net"] = get_prob_net(None,snn,input_shape=(2,128,128))
        data["IBMGestures_session_id"] = sid
        return data

class NMNIST:
    @staticmethod
    def make():
        d = NMNIST.default_hyperparameters()

        def mk_data_dir(mode="direct"):
            if mode == "direct":
                return "data/N-MNIST/"
            elif mode == "bsub":
                return "$SCRATCH/data/N-MNIST/"
            raise Exception("Invalid Mode")

        d["mk_data_dir"] = mk_data_dir
        d["data_dir"] = "{mk_data_dir({mode})}"
        d["code_file"] = "main_NMNIST.py"
        d["architecture"] = "NMNIST"
        d["train"] = mk_runner(NMNIST, ["data_dir"])
        return d

    @staticmethod
    def default_hyperparameters():
        d = standard_defaults()
        return d

    @staticmethod
    def checker(sid, table, cache_dir):
        return True

    @staticmethod
    def get_flags():
        default_dict = {
            **NMNIST.default_hyperparameters(),
            **{"data_dir": "data/N-MNIST/"},
        }
        return _get_flags(default_dict, help())

    @staticmethod
    def loader(sid, table, cache_dir):
        data = {}
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, f"Resources/Models/{sid}_model.pt")
        ann = load_ann(model_path)
        snn = get_summed_network(ann, n_classes=10)
        prob_net = get_prob_net(ann,snn)
        data["ann"] = ann
        data["snn"] = snn
        data["prob_net"] = prob_net
        data["NMNIST_session_id"] = sid
        return data


class BMNIST:
    @staticmethod
    def make():
        d = BMNIST.default_hyperparameters()

        def mk_data_dir(mode="direct"):
            if mode == "direct":
                return "data/B-MNIST/"
            elif mode == "bsub":
                return "$SCRATCH/data/B-MNIST/"
            raise Exception("Invalid Mode")

        d["mk_data_dir"] = mk_data_dir
        d["data_dir"] = "{mk_data_dir({mode})}"
        d["code_file"] = "main_BMNIST.py"
        d["architecture"] = "BMNIST"
        d["train"] = mk_runner(BMNIST, ["data_dir"])
        return d

    @staticmethod
    def default_hyperparameters():
        d = standard_defaults()
        return d

    @staticmethod
    def checker(sid, table, cache_dir):
        return True

    @staticmethod
    def get_flags():
        default_dict = {
            **BMNIST.default_hyperparameters(),
            **{"data_dir": "data/B-MNIST/"},
        }
        return _get_flags(default_dict, help())

    @staticmethod
    def loader(sid, table, cache_dir):
        data = {}
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, f"Resources/Models/{sid}_model.pt")
        ann = load_ann(model_path, ann=get_mnist_ann_arch())
        prob_net = get_prob_net_continuous(ann=ann)
        data["ann"] = ann
        data["prob_net"] = prob_net
        data["BMNIST_session_id"] = sid
        return data
