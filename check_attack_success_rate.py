import numpy as np
import os
import re


if __name__ == "__main__":
    data_dir = "./attack_result_csv"
    csv_file_list = [each for each in os.listdir(data_dir) if each.endswith(".csv")]
    for csv_file in csv_file_list:
        numbers = re.findall(r"\d+\.?\d*", csv_file)
        epoch = int(numbers[0])
        target_label = int(numbers[1])
        try:
            with open(os.path.join(data_dir, csv_file)) as f:
                lines = f.readlines()
                success_sample_targeted_chip = 0  # number of success attack testing samples (on-chip)
                success_sample_random_chip = 0
                success_sample_targeted_sim = 0  # number of success attack testing samples (simulated)
                success_sample_random_sim = 0
                total_sample = 0
                for idx, line in enumerate(lines):
                    if idx == 0: continue
                    # get the success samples numbers and total valid sample numbers
                    data = re.findall(r"\d+\.?\d*", line)
                    data = [int(each) for each in data]
                    _, ground_truth, chip_out, sim_out, chip_out_attacked_targeted, chip_out_attacked_random, sim_out_attacked_targeted, sim_out_attacked_random = data
                    if ground_truth == target_label: continue
                    if chip_out_attacked_targeted == target_label: success_sample_targeted_chip += 1
                    if chip_out_attacked_random == target_label: success_sample_random_chip += 1
                    if sim_out_attacked_targeted == target_label: success_sample_targeted_sim += 1
                    if sim_out_attacked_random == target_label: success_sample_random_sim += 1

                    total_sample += 1
                success_rate_targeted_chip = success_sample_targeted_chip / total_sample
                success_rate_random_chip = success_sample_random_chip / total_sample
                success_rate_targeted_sim = success_sample_targeted_sim / total_sample
                success_rate_random_sim = success_sample_random_sim / total_sample

                print(f"Epoch:{epoch}| Label: {target_label}"
                      f"| Target Patch Success Rate(on-chip): {success_rate_targeted_chip}"
                      f"| Random Patch Success Rate(on-chip): {success_rate_random_chip}"
                      f"| Target Patch Success Rate(Simulate): {success_rate_targeted_sim}"
                      f"| Random Patch Success Rate(Simulate):{success_rate_random_sim}\n")
        except:
            Warning("File still not valid")
