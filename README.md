# graspRRT
This repository contains a simplified implementation of the algorithm used by the MIT-Princeton team at the 2017
[Amazon Robotics Challenge](https://vision.princeton.edu/projects/2017/arc/). 

### Installation
There are three different .yaml files provided to install all dependencies related to this project.
```bash
# dependencies for ICP pose estimation
mamba env create -f env_icp.yaml

# dependencies for visual affordance on cpu/mps
mamba env create -f env_visual_cpu.yaml

# dependencies for visual affordance on gpu
mamba env create -f env_visual_gpu.yaml
```
### ICP Pose Estimation
In order to run the Amazon grasp challenege using a ICP pose estimation with RRT (rapidly exploring random tree),
run the file file <strong>clear_bin.py</strong> with the following arguments:
```bash
cd icp_src/
python clear_bin.py -disp
```
Please note depending on the build version of OpenGL and PyBullet, there may be some display artifacts when running the simulation.

### Visual Affordance
The visual affordance folder (visual_src) contains the best network weights for performing visual affordance.
To test the visual affordance model, run the following:
```bash
cd visual_src
python eval.py --model action_regression --task empty_bin --seed 5
```


