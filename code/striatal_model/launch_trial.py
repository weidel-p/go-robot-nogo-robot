import os
import sys
import time
import subprocess
import signal
import rosbag
import json
from subprocess import check_output

experiment = sys.argv[3]
scale = float(sys.argv[6])
trial = sys.argv[9]

if scale == 1.:
    folder_ = "data"
else:
    folder_ = "data_long"

print experiment, scale, trial


def kill_ros():
    os.system("kill $(pgrep ros)")
    time.sleep(20.)


def start_ros():
    os.system("roslaunch pioneer3at empty.launch &")
    time.sleep(10.)
    os.system("rosparam set \use_sim_time true")


def record_odom():
    sp = subprocess.Popen(["python", "../helper/record_movement.py"])
    #sp = rosbag.rosbag_main.record_cmd(["-O", "odom", "/odom"])
    return sp


def run_simulation():
    sp = subprocess.Popen(["mpirun", "-np", "8", "music", "main.music"])
    return sp


def kill(name):
    print "kill", name
    pid = check_output(["pidof", name])
    os.kill(int(pid), signal.SIGINT)
    time.sleep(5.)


def kill_pid(pid):
    print "kill", pid
    os.kill(int(pid), signal.SIGINT)
    time.sleep(5.)


def move_results(session):
    print folder_
    print experiment
    print trial
    os.system("mv odom.bag ../../{}/{}/{}/".format(folder_, experiment, trial))
    os.system("mv ../../{}/left_hemisphere-* ../../{}/{}/{}/".format(folder_,
                                                                     folder_, experiment, trial))
    os.system("mv ../../{}/right_hemisphere-* ../../{}/{}/{}/".format(folder_,
                                                                      folder_, experiment, trial))
    os.system("mv ../../{}/neuron_ids* ../../{}/{}/{}/".format(folder_,
                                                               folder_, experiment, trial))


def run(num_trials):
    for s in range(num_trials):
        print("Running create_voting_readout.py. to regenerate channels_readout.dat")
        os.system("python create_voting_readout.py")

        print("starting session {}".format(s))
        start_ros()

        path = (os.environ['ROS_PACKAGE_PATH']).split(':')[0]

        print path

        os.system("cp experiments/{} cfg.yaml".format(experiment))

        with open("scale.json", "w") as f:
            json.dump({"scale": scale}, f)

        with open("trial.json", "w") as f:
            json.dump({"trial": trial}, f)

        os.system(
            "rosrun gazebo_ros spawn_model -sdf -file {}/pioneer3at/sdf/pioneer3at.sdf -model pioneer3at -x 0 -y 0".format(path))
        time.sleep(5)

        os.system("rosparam set \use_sim_time true")
        os.system("gz physics -s " + str(2.0 / 1000))
        rec = record_odom()
        sim = run_simulation()

        # sleep for the simtime (10/ 0.15) plus one minute for saving and loading data etc
        #time.sleep(10 + 20 / 0.5)
        time.sleep(20 + (20 * scale) / 0.2)

        kill("record")
        kill_pid(sim.pid)
        kill_ros()

        os.system("rm cfg.yaml")
        os.system("rm scale.json")
        os.system("rm trial.json")

        move_results(s)
        print "move done !"


run(1)
