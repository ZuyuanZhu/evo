import os
import json
import numpy as np
from evo.core import trajectory, sync, metrics
from evo.tools import file_interface, plot


class APECalculator:
    def __init__(self, base_loc, traj_ref_file, trajA_est_file, trajB_est_file, trajB_est_file_cont):
        self.base_loc = base_loc
        self.traj_ref_file = traj_ref_file
        self.trajA_est_file = trajA_est_file
        self.trajB_est_file = trajB_est_file
        self.trajB_est_file_cont = trajB_est_file_cont

        self.traj_ref_A = None
        self.traj_ref_B = None
        self.traj_est_A = None
        self.traj_est_B = None
        self.traj_est_B_time = None

        self.ape_metric_A = None
        self.ape_metric_B = None

    def load_trajectories(self):
        traj_ref_file_ = os.path.join(self.base_loc, self.traj_ref_file)
        trajA_est_file_ = os.path.join(self.base_loc, self.trajA_est_file)
        trajB_est_file_ = os.path.join(self.base_loc, self.trajB_est_file)
        trajB_est_file_time_ = os.path.join(self.base_loc, self.trajB_est_file_cont)

        self.traj_ref_A = file_interface.read_tum_trajectory_file(traj_ref_file_)
        self.traj_ref_B = file_interface.read_tum_trajectory_file(traj_ref_file_)
        self.traj_est_A = file_interface.read_tum_trajectory_file(trajA_est_file_)
        self.traj_est_B = file_interface.read_tum_trajectory_file(trajB_est_file_)
        self.traj_est_B_time = file_interface.read_tum_trajectory_file(trajB_est_file_time_)

    def register_and_align_trajectories(self):
        self.traj_ref_A, self.traj_est_A = sync.associate_trajectories(self.traj_ref_A, self.traj_est_A)
        self.traj_ref_B, self.traj_est_B = sync.associate_trajectories(self.traj_ref_B, self.traj_est_B)
        self.traj_est_A.align(self.traj_ref_A, correct_scale=True)
        self.traj_est_B.align(self.traj_ref_B, correct_scale=True)

    def calculate_ape(self):
        data_A = (self.traj_ref_A, self.traj_est_A)
        data_B = (self.traj_ref_B, self.traj_est_B)
        self.ape_metric_A = metrics.APE(metrics.PoseRelation.translation_part)
        self.ape_metric_B = metrics.APE(metrics.PoseRelation.translation_part)

        self.ape_metric_A.process_data(data_A)
        self.ape_metric_B.process_data(data_B)

    def calculate_combined_ape(self):
        ape_statistics_A = self.ape_metric_A.get_all_statistics()
        ape_statistics_B = self.ape_metric_B.get_all_statistics()

        # remove sse
        ape_statistics_without_sse_A = ape_statistics_A.pop('sse')
        ape_statistics_without_sse_B = ape_statistics_B.pop('sse')

        print("A mean:", ape_statistics_A["mean"])
        print("B mean:", ape_statistics_B["mean"])

        # calculate combined APE
        ape_statistics_AB, ape_metric_error_AB = self.combined_ape(
            self.traj_est_A.positions_xyz, self.traj_est_B.positions_xyz,
            self.traj_ref_A.positions_xyz, self.traj_ref_B.positions_xyz)

        # save the ape_statistics_AB to local file
        results_loc = os.path.join(self.base_loc, 'results')
        if not os.path.exists(results_loc):
            os.makedirs(results_loc)
        with open(os.path.join(results_loc, "agentAB_ape_cont_statistics.csv"), "w") as file:
            # write the dictionary to the file in JSON format
            json.dump('APE statistics for combined trajectory AB: ', file)
            json.dump(ape_statistics_AB, file)

            return ape_statistics_AB, ape_metric_error_AB

    @staticmethod
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

    def plot_ape(self, ape_statistics_AB, ape_metric_error_AB):
        print("loading plot modules")
        import matplotlib.pyplot as plt

        print("plotting")
        plot_collection = plot.PlotCollection("Example")
        # metric values
        fig_1 = plt.figure(figsize=(6.2, 3.5))

        # do not plot sse
        ape_statistics_AB.pop('sse')
        timestamp = np.append(self.traj_est_A.timestamps, self.traj_est_B_time.timestamps)
        plot.error_array(fig_1.gca(), ape_metric_error_AB, timestamp, statistics=ape_statistics_AB,
                         name="APE", title="Combined APE")

        plot_collection.add_figure("raw", fig_1)

        # trajectory colormapped with error
        fig_2 = plt.figure(figsize=(7, 7))
        plot_mode = plot.PlotMode.xz
        ax = plot.prepare_axis(fig_2, plot_mode)
        plot.traj(ax, plot_mode, self.traj_ref_A, '--', 'gray', 'reference')
        plot.traj(ax, plot_mode, self.traj_ref_B, '--', 'gray')

        # calculate the minimum and maximum values for the color map
        min_ape = min(self.ape_metric_A.error.min(), self.ape_metric_B.error.min())
        max_ape = max(self.ape_metric_A.error.max(), self.ape_metric_B.error.max())

        plot.traj_colormap(ax, self.traj_est_A, self.ape_metric_A.error, plot_mode,
                           min_map=min_ape,
                           max_map=max_ape,
                           title="APE mapped onto trajectory")
        plot.traj_colormap(ax, self.traj_est_B, self.ape_metric_B.error, plot_mode,
                           min_map=min_ape,
                           max_map=max_ape,
                           fig=plt.gcf())
        plot_collection.add_figure("traj (error)", fig_2)

        plot_collection.show()


if __name__ == "__main__":
    base_loc = "/home/zuyuan/Documents/dataset/kitti/Traj_Vis/Self_trained_ORBvoc_seq00/seq02_2-3.5_12-13/"
    traj_ref_file = "02_time_poses.txt"
    trajA_est_file = "KF_GBA_0_sorted.csv"
    trajB_est_file = "KF_GBA_1_sorted.csv"
    trajB_est_file_cont = "KF_GBA_1_sorted_cont.csv"

    calculator = APECalculator(base_loc, traj_ref_file, trajA_est_file, trajB_est_file, trajB_est_file_cont)
    calculator.load_trajectories()
    calculator.register_and_align_trajectories()
    calculator.calculate_ape()
    ape_statistics_AB, ape_metric_error_AB = calculator.calculate_combined_ape()
    calculator.plot_ape(ape_statistics_AB, ape_metric_error_AB)
