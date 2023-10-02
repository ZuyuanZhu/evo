import rosbag
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
import sys


def extract_odometry_data(bag_file, out_tum_traj, start_time):
    with rosbag.Bag(bag_file, 'r') as bag:
        # Extract the first timestamp
        first_timestamp = None
        for topic, msg, t in bag.read_messages(topics=['/zed2/zed_node_test/odom']):
            first_timestamp = msg.header.stamp.to_sec()
            break

        if first_timestamp is None:
            print("No messages found!")
            return

        time_offset = start_time - first_timestamp

        with open(out_tum_traj, 'w') as outfile:
            for topic, msg, t in bag.read_messages(topics=['/zed2/zed_node_test/odom']):

                adjusted_timestamp = msg.header.stamp.to_sec() + time_offset
                position = msg.pose.pose.position
                orientation = msg.pose.pose.orientation

                data = [adjusted_timestamp, position.x, position.y, position.z,
                        orientation.x, orientation.y, orientation.z, orientation.w]

                outfile.write(" ".join(map(str, data)) + "\n")
    print(f"TUM traj saved to {out_tum_traj}")


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python script_name.py path_to_rosbag")
    #     sys.exit(1)
    #
    # bag_file = sys.argv[1]

    base = "/media/zuyuan/DATA1TB/Jackal/bags_data_campaign_july_2023/"
    group_folder = "rosbag_2_2_upper_half"

    in_bag = "test_2_10_55_select_frame_id_odom"
    start_time = 10

    bag_file = base + in_bag + ".bag"
    out_base = "/media/zuyuan/DATA1TB/Jackal/bags_data_campaign_july_2023/Evaluation/Zed_Odom_replace_track_pose_estimation/"
    out_tum_traj = out_base + group_folder + "/" + in_bag + "_TUM_traj.csv"
    extract_odometry_data(bag_file, out_tum_traj, start_time)
