import os


if __name__ == "__main__":
    patch_size = "0.035"
    for epoch in [10]:
        for label in range(0, 11):
            os.system(f"python dynapcnn_test_use_patch.py {label} {epoch} {patch_size.split('.')[1]}")
