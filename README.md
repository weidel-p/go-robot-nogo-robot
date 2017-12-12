# go-robot-nogo-robot

### Installation

#### Dependencies:

1. NEST 2.12 (http://nest-simulator.org/)
2. MUSIC (https://github.com/INCF/MUSIC)
3. MUSIC adapters (https://github.com/incf-music/music-adapters)
4. ROS/Gazebo Kinetic (http://www.ros.org/)
5. Snakemake (http://snakemake.readthedocs.io/en/stable/index.html)

### Usage:

Clone this repository and run following instructions.
> git clone git@github.com:weidel-p/go-robot-nogo-robot.git

First get all submodules.
> git submodule init
> git submodule update --remote

Then build the robotic environment.
> cd catkin_ws
> catkin_make
> catkin_make install
> source devel/setup.zsh
> cd ..

Run the model and plot all figures (this may take a while).
> snakemake






