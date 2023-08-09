
"""
To convert the GPS and IMU information from a ROS bag file into the ORB-SLAM2
trajectory format, follow these steps:

1. Extract GPS and IMU data from the ROS bag.
2. Convert GPS data to a local Cartesian frame.
3. Extract and align the orientation data from the IMU.
4. Store the resulting data in the desired format.
"""

import rosbag
import utm
import numpy as np
import tf.transformations as tf_trans


def get_local_position(gps_origin, lat, lon):
    x, y, _, _ = utm.from_latlon(lat, lon)
    dx = x - gps_origin[0]
    dy = y - gps_origin[1]
    return dx, dy


def get_orientation_from_imu(imu_msg):
    orientation = imu_msg.orientation
    return [orientation.x, orientation.y, orientation.z, orientation.w]



# Read ROS bag
base_path = "/media/zuyuan/DATA1TB/Jackal/bags_data_campaign_july_2023/"
bag_name = "test_2_2023-07-04-10-13-18"
out_name = "groundtruth.txt"
out_path = base_path + "groundtruth/"
bag = rosbag.Bag(base_path + bag_name + ".bag")
gps_origin = None
groundtruth_data = []
gps_data = []
imu_data = []


for topic, msg, t in bag.read_messages(topics=['/sugv/emlid_rtk_rover/tcpfix', '/sugv/imu/data']):
    if topic == '/sugv/emlid_rtk_rover/tcpfix':
        if gps_origin is None:
            gps_origin = utm.from_latlon(msg.latitude, msg.longitude)[:2]

        x, y = get_local_position(gps_origin, msg.latitude, msg.longitude)
        z = msg.altitude  # or msg.altitude if available
        gps_data.append((msg.header.stamp.to_sec(), x, y, z))

    elif topic == '/sugv/imu/data':
        qx, qy, qz, qw = get_orientation_from_imu(msg)
        imu_data.append((msg.header.stamp.to_sec(), qx, qy, qz, qw))

bag.close()

"""
 For the rosbag 2, 
 the IMU publishes at approximately 6199/258≈218 messages per second (Hz), 
 whereas the GPS publishes at about 2583/258≈10 messages per second (Hz). 
 The GPS is publishing at a much lower rate than the IMU.
 
 Finding the closest IMU message for each GPS message without 
 unnecessarily skipping many IMU messages.
"""

# Threshold for the timestamp difference (in seconds)
TIME_DIFF_THRESHOLD = 0.05  # 50 milliseconds

# Merge GPS and IMU data based on timestamps
groundtruth_data = []
imu_index = 0

for gps_time, x, y, z in gps_data:
    # Find the closest IMU data to the current GPS data
    closest_diff = float('inf')
    closest_imu_data = None

    for imu_time, qx, qy, qz, qw in imu_data:
        time_diff = abs(gps_time - imu_time)

        if time_diff < closest_diff:
            closest_diff = time_diff
            closest_imu_data = (imu_time, qx, qy, qz, qw)

    # Check if timestamp difference is within an acceptable range
    if closest_diff <= TIME_DIFF_THRESHOLD:
        groundtruth_data.append([gps_time, x, y, z] + list(closest_imu_data[1:]))
    else:
        print(f"Skipped data at timestamp {gps_time} due to exceeding time difference.")

# Save data
with open(out_path+out_name, 'w') as f:
    for data in groundtruth_data:
        f.write(' '.join(map(str, data)) + '\n')

print("Done!")
