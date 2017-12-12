import rosbag

rosbag.rosbag_main.record_cmd(["-O", "odom", "/odom"])
