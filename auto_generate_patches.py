import os

"""
Generate adversarial patches by different hyper-param(epoch & target_label)
"""


if __name__ == "__main__":
    patch_size = 0.025
    for label in range(0, 11):
        for epoch in [10]:
            os.system(f"python dynapcnn_generate_patches.py {epoch} {label} {patch_size}")