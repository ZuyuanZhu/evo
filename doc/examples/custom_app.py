#!/usr/bin/env python

print("loading required evo modules")
from evo.core import trajectory, sync, metrics
from evo.tools import file_interface

print("loading trajectories")
# traj_ref = file_interface.read_tum_trajectory_file(
#     "../../test/data/fr2_desk_groundtruth.txt")
# traj_est = file_interface.read_tum_trajectory_file(
#     "../../test/data/fr2_desk_ORB.txt")

base_loc = "/home/zuyuan/rasberry_ws/src/data_kitti/OUTPUT/Traj_Vis/seq00_agentA0-320_agentB1378-1698_test/"
trajA_est_file = base_loc + "KF_GBA_0_sorted.csv"
traj_ref_file = base_loc + "00_time_poses.txt"

trajB_est_file = base_loc + "KF_GBA_1_sorted.csv"

traj_ref_A = file_interface.read_tum_trajectory_file(traj_ref_file)
traj_ref_B = file_interface.read_tum_trajectory_file(traj_ref_file)
traj_est_A = file_interface.read_tum_trajectory_file(trajA_est_file)
traj_est_B = file_interface.read_tum_trajectory_file(trajB_est_file)

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
fig_1 = plt.figure(figsize=(16, 16))
plot.error_array(fig_1.gca(), ape_metric_A.error, statistics=ape_statistics_A,
                 name="APE", title=str(ape_metric_A))
plot.error_array(fig_1.gca(), ape_metric_B.error, statistics=ape_statistics_B,
                 name="APE", title=str(ape_metric_B))
plot_collection.add_figure("raw", fig_1)

# trajectory colormapped with error
fig_2 = plt.figure(figsize=(16, 16))
plot_mode = plot.PlotMode.xz
ax = plot.prepare_axis(fig_2, plot_mode)
plot.traj(ax, plot_mode, traj_ref_A, '--', 'gray', 'reference')
plot.traj(ax, plot_mode, traj_ref_B, '--', 'gray', 'reference')

plot.traj_colormap(ax, traj_est_A, ape_metric_A.error, plot_mode,
                   min_map=ape_statistics_A["min"],
                   max_map=ape_statistics_A["max"],
                   title="APE mapped onto trajectory")
plot.traj_colormap(ax, traj_est_B, ape_metric_B.error, plot_mode,
                   min_map=ape_statistics_B["min"],
                   max_map=ape_statistics_B["max"],
                   title="APE mapped onto trajectory")
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
