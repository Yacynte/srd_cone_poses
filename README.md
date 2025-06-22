# srd_cone_poses
Calculating Cone Poses from Stereo Camera System

## Clone and Build

First, clone the repository including its submodules:

```bash
git clone git@github.com:Yacynte/srd_cone_poses.git
cd srd_cone_poses
```
Build the package using `colcon`:

```bash
colcon build --packages-select keypoint_sorting
source install/setup.sh
ros2 run keypoint_sorting keypointClosesetNeighboor
```
