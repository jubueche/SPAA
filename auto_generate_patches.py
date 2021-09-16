import os

"""
Generate adversarial patches by different hyper-param(epoch & target_label)
"""


if __name__ == "__main__":
    for epoch in [1, 2, 3, 5, 7, 10]:
        for label in range(2):
            os.system(f"python dynapcnn_generate_patches.py {epoch} {label}")