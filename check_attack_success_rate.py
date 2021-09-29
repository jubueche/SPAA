import numpy as np
import os
import re
import csv


if __name__ == "__main__":
    data_dir = "./attack_result_csv"
    csv_file_list = [each for each in os.listdir(data_dir) if each.endswith(".csv")]
    results = []
    for csv_file in csv_file_list:
        numbers = re.findall(r"\d+\.?\d*", csv_file)
        epoch = int(numbers[0])
        target_label = int(numbers[1])
        try:
            with open(os.path.join(data_dir, csv_file)) as f:
                lines = f.readlines()
                consistent_prediction_sim_chip = 0
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
                    if chip_out == sim_out: consistent_prediction_sim_chip += 1
                    if chip_out_attacked_targeted == target_label: success_sample_targeted_chip += 1
                    if chip_out_attacked_random == target_label: success_sample_random_chip += 1
                    if sim_out_attacked_targeted == target_label: success_sample_targeted_sim += 1
                    if sim_out_attacked_random == target_label: success_sample_random_sim += 1

                    total_sample += 1
                consistent_prediction_sim_chip = round(consistent_prediction_sim_chip / total_sample, 4)
                success_rate_targeted_chip = round(success_sample_targeted_chip / total_sample, 4)
                success_rate_random_chip = round(success_sample_random_chip / total_sample, 4)
                success_rate_targeted_sim = round(success_sample_targeted_sim / total_sample, 4)
                success_rate_random_sim = round(success_sample_random_sim / total_sample, 4)
                
                results.append((int(numbers[1]), consistent_prediction_sim_chip, success_rate_targeted_chip, success_rate_random_chip, success_rate_targeted_sim, success_rate_random_sim))
                print(f"Epoch:{epoch}| Label: {target_label}"
                      f"| Target Patch Success Rate(on-chip): {success_rate_targeted_chip}"
                      f"| Random Patch Success Rate(on-chip): {success_rate_random_chip}"
                      f"| Target Patch Success Rate(Simulate): {success_rate_targeted_sim}"
                      f"| Random Patch Success Rate(Simulate):{success_rate_random_sim}\n")
        except:
            Warning("File still not valid")

    f = open(data_dir+f'/ep{numbers[0]}_lbALL_num{numbers[2]}_patchsize{numbers[3]}.csv', 'w')
    writer = csv.writer(f)

    writer.writerow(("Label", "Simulation/Chip consistency on original prediction", "Target Patch Success Rate(on-chip)", "Random Patch Success Rate(on-chip)",\
                     "Target Patch Success Rate(Simulate)", "Random Patch Success Rate(Simulate)"))
    
    sorted_by_label = sorted(results, key=lambda tup: tup[0])
    for result in sorted_by_label:
        writer.writerow(result)
    f.close()