#!/usr/bin/env python

print("loading required evo modules")
from evo.core import trajectory, sync, metrics
from evo.tools import file_interface
import numpy as np


def combined_ape(traj_A, traj_B, ref_A, ref_B):

    # Merge the two trajectories A and B into a single trajectory C
    trajectory_AB = np.concatenate((traj_A, traj_B), axis=0)
    ref_AB = np.concatenate((ref_A, ref_B), axis=0)

    # Calculate the APE for each pose in trajectory C
    ape_C = np.linalg.norm(trajectory_AB - ref_AB, axis=1)

    # Compute the statistics of the APE values for trajectory C
    ape_statistics_AB = {
        'rmse': np.sqrt(np.mean(ape_C ** 2)),
        'mean': np.mean(ape_C),
        'median': np.median(ape_C),
        'std': np.std(ape_C),
        'min': np.min(ape_C),
        'max': np.max(ape_C),
        'sse': np.sum(ape_C ** 2)
    }

    print("APE statistics for combined trajectory AB:", ape_statistics_AB)
    return ape_statistics_AB, ape_C



print("loading trajectories")
# traj_ref = file_interface.read_tum_trajectory_file(
#     "../../test/data/fr2_desk_groundtruth.txt")
# traj_est = file_interface.read_tum_trajectory_file(
#     "../../test/data/fr2_desk_ORB.txt")

base_loc = "/home/zuyuan/Documents/dataset/kitti/Traj_Vis/seq00_agentA0-320_agentB1378-1698_test/"
trajA_est_file = base_loc + "KF_GBA_0_sorted.csv"
traj_ref_file = base_loc + "00_time_poses.txt"

trajB_est_file = base_loc + "KF_GBA_1_sorted.csv"
trajB_est_file_time = base_loc + "KF_GBA_1_sorted_cont.csv"

traj_ref_A = file_interface.read_tum_trajectory_file(traj_ref_file)
traj_ref_B = file_interface.read_tum_trajectory_file(traj_ref_file)
traj_est_A = file_interface.read_tum_trajectory_file(trajA_est_file)
traj_est_B = file_interface.read_tum_trajectory_file(trajB_est_file)
traj_est_B_time = file_interface.read_tum_trajectory_file(trajB_est_file_time)


print("registering and aligning trajectories")
traj_ref_A, traj_est_A = sync.associate_trajectories(traj_ref_A, traj_est_A)
traj_ref_B, traj_est_B = sync.associate_trajectories(traj_ref_B, traj_est_B)
traj_est_A.align(traj_ref_A, correct_scale=True)
traj_est_B.align(traj_ref_B, correct_scale=True)

print("calculating APE")
data_A = (traj_ref_A, traj_est_A)
data_B = (traj_ref_B, traj_est_B)
ape_metric_A = metrics.APE(metrics.PoseRelation.translation_part)
ape_metric_B = metrics.APE(metrics.PoseRelation.translation_part)

ape_metric_A.process_data(data_A)
ape_metric_B.process_data(data_B)
ape_statistics_A = ape_metric_A.get_all_statistics()
ape_statistics_B = ape_metric_B.get_all_statistics()

# remove sse
ape_statistics_without_sse_A = ape_statistics_A.pop('sse')
ape_statistics_without_sse_B = ape_statistics_B.pop('sse')

print("A mean:", ape_statistics_A["mean"])
print("B mean:", ape_statistics_B["mean"])

print("loading plot modules")
from evo.tools import plot
import matplotlib.pyplot as plt

print("plotting")
plot_collection = plot.PlotCollection("Example")
# metric values
fig_1 = plt.figure(figsize=(6.2, 3.5))

# ape_metric_error_AB = np.append(ape_metric_A.error, ape_metric_B.error)
# ape_statistics_AB = {}
# ape_statistics_AB['rmse'] = (ape_statistics_A['rmse'] + ape_statistics_B['rmse'])/2
# ape_statistics_AB['mean'] = (ape_statistics_A['mean'] + ape_statistics_B['mean'])/2
# ape_statistics_AB['median'] = (ape_statistics_A['median'] + ape_statistics_B['median'])/2
# ape_statistics_AB['std'] = (ape_statistics_A['std'] + ape_statistics_B['std'])/2
# ape_statistics_AB['min'] = min(ape_statistics_A['min'], ape_statistics_B['min'])
# ape_statistics_AB['max'] = max(ape_statistics_A['max'], ape_statistics_B['max'])
ape_metric_AB = ape_metric_A

ape_statistics_AB, ape_metric_error_AB = combined_ape(traj_est_A.positions_xyz, traj_est_B.positions_xyz, traj_ref_A.positions_xyz, traj_ref_B.positions_xyz)
ape_statistics_AB.pop('sse')
# print(f"ape_statistics_AB: {ape_statistics_AB}")
# plot.error_array(fig_1.gca(), ape_metric_A.error, statistics=ape_statistics_A,
#                  name="APE", title=str(ape_metric_A))
# plot.error_array(fig_1.gca(), ape_metric_B.error, statistics=ape_statistics_B,
#                  name="APE", title=str(ape_metric_B))
timestamp = np.append(traj_est_A.timestamps, traj_est_B_time.timestamps)
plot.error_array(fig_1.gca(), ape_metric_error_AB, timestamp, statistics=ape_statistics_AB,
                 name="APE", title=str(ape_metric_AB))

plot_collection.add_figure("raw", fig_1)

# trajectory colormapped with error
fig_2 = plt.figure(figsize=(7, 7))
plot_mode = plot.PlotMode.xz
ax = plot.prepare_axis(fig_2, plot_mode)
plot.traj(ax, plot_mode, traj_ref_A, '--', 'gray', 'reference')
plot.traj(ax, plot_mode, traj_ref_B, '--', 'gray')

# calculate the minimum and maximum values for the color map
min_ape = min(ape_statistics_A["min"], ape_statistics_B["min"])
max_ape = max(ape_statistics_A["max"], ape_statistics_B["max"])

plot.traj_colormap(ax, traj_est_A, ape_metric_A.error, plot_mode,
                   min_map=min_ape,
                   max_map=max_ape,
                   title="APE mapped onto trajectory")
plot.traj_colormap(ax, traj_est_B, ape_metric_B.error, plot_mode,
                   min_map=min_ape,
                   max_map=max_ape,
                   fig=plt.gcf())
plot_collection.add_figure("traj (error)", fig_2)

# # trajectory colormapped with speed
# fig_3 = plt.figure(figsize=(8, 8))
# plot_mode = plot.PlotMode.xy
# ax = plot.prepare_axis(fig_3, plot_mode)
# speeds = [
#     trajectory.calc_speed(traj_est.positions_xyz[i],
#                           traj_est.positions_xyz[i + 1],
#                           traj_est.timestamps[i], traj_est.timestamps[i + 1])
#     for i in range(len(traj_est.positions_xyz) - 1)
# ]
# speeds.append(0)
# plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
# plot.traj_colormap(ax, traj_est, speeds, plot_mode, min_map=min(speeds),
#                    max_map=max(speeds), title="speed mapped onto trajectory")
# fig_3.axes.append(ax)
# plot_collection.add_figure("traj (speed)", fig_3)

plot_collection.show()



