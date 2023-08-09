#!/usr/bin/env python
from evo.tools.file_interface import read_tum_trajectory_file, write_tum_trajectory_file


base_path = "/media/zuyuan/DATA1TB/Jackal/bags_data_campaign_july_2023/Evaluation/rosbag_2_1_upper_half/"
agentA_traj_sort = base_path + "KF_GBA_0_sorted.csv"
agentB_traj_sort = base_path + "KF_GBA_1_sorted.csv"
groundTruth_orig = base_path + "rosbag_2_groundtruth.txt"

agentA_traj = read_tum_trajectory_file(agentA_traj_sort)
agentB_traj = read_tum_trajectory_file(agentB_traj_sort)
groundTruth = read_tum_trajectory_file(groundTruth_orig)

first_timestamp_A = agentA_traj.timestamps[0]
agentA_traj.timestamps -= first_timestamp_A
agentA_traj.timestamps += 10

# use relative time, 10-55
write_tum_trajectory_file(base_path + "KF_GBA_0_sorted_10_55.csv", agentA_traj)

# convert ground truth, use relative timestamps
first_timestamp_GT = groundTruth.timestamps[0]
groundTruth.timestamps -= first_timestamp_GT
write_tum_trajectory_file(base_path + "rosbag_2_groundtruth_relative.csv", groundTruth)

last_timestamp_A = agentA_traj.timestamps[-1]
first_timestamp_B = agentB_traj.timestamps[0]

# convert ground truth, use relative timestamps
agentB_traj.timestamps -= first_timestamp_B
agentB_traj.timestamps += 127
write_tum_trajectory_file(base_path + "KF_GBA_1_sorted_127_167.csv", agentB_traj)
print("")
