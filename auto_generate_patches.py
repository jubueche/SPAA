import os

"""
Generate adversarial patches by different hyper-param(epoch & target_label)
"""


if __name__ == "__main__":
    for label in range(6):
        for epoch in [1, 3, 5, 10]:
            os.system(f"python dynapcnn_generate_patches.py {epoch} {label}")