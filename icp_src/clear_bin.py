import sim
import os
import camera
import pybullet as p
import numpy as np
import torch
import train_seg_model
import torchvision
import icp
import transforms
from scipy.spatial.transform import Rotation
import random
from rrt import *
import argparse
import pdb

if __name__ == "__main__":
    if not os.path.exists('checkpoint_multi.pth.tar'):
        print("Error: 'checkpoint_multi.pth.tar' not found.")
        exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-disp', action='store_true')
    args = parser.parse_args()

    random.seed(1)
    color_palette = train_seg_model.get_tableau_palette()

    object_shapes = [
        "assets/objects/cube.urdf",
        "assets/objects/rod.urdf",
        "assets/objects/custom.urdf",
    ]
    object_meshes = [
        "assets/objects/cube.obj",
        "assets/objects/rod.obj",
        "assets/objects/custom.obj",
    ]
    env = sim.PyBulletSim(object_shapes = object_shapes, gui=args.disp)
    env.load_gripper()

    my_camera = camera.Camera(
        image_size=(480, 640),
        near=0.01,
        far=10.0,
        fov_w=50
    )
    camera_target_position = (env._workspace1_bounds[:, 0] + env._workspace1_bounds[:, 1]) / 2
    camera_target_position[2] = 0
    camera_distance = np.sqrt(((np.array([0.5, -0.5, 0.8]) - camera_target_position)**2).sum())
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target_position,
        distance=camera_distance,
        yaw=90,
        pitch=-60,
        roll=0,
        upAxisIndex=2,
    )

    # Prepare model (again, should be consistent with segmentation training)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    n_channels = 3  # RGB
    n_classes = len(object_shapes) + 1  # number of objects + 1 for background class
    model = train_seg_model.miniUNet(n_channels, n_classes)
    model.to(device)
    model, _, _ = train_seg_model.load_chkpt(model, 'checkpoint_multi.pth.tar', device)
    model.eval()

    rgb_trans = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(train_seg_model.mean_rgb, train_seg_model.std_rgb),
    ])

    obj_ids = env._objects_body_ids 

    is_grasped = np.zeros(3).astype(np.bool)
    while not np.all(is_grasped):
        rgb_obs, depth_obs, _ = camera.make_obs(my_camera, view_matrix)
        rgb_tensor = rgb_trans(rgb_obs).unsqueeze(0)
        output = model(rgb_tensor.to(device)) # pred should contain the predicted segmentation mask
        _, tensor_mask = torch.max(output, dim = 1)
        pred_mask = tensor_mask.cpu().detach().numpy()[0]

        markers = []
        num_sample_pts = 1000

        # Randomly choose an object index to grasp which is not grasped yet.
        obj_index = np.random.choice(np.where(~is_grasped)[0], 1)[0]

        obj_depth = np.zeros_like(depth_obs)
        w, h = obj_depth.shape
        for i in range(w):
            for j in range(h):
                if pred_mask[i,j] == (obj_index+1):
                    obj_depth[i,j] = depth_obs[i,j]

        cam_pts = np.zeros((0,3))
        cam_pts = np.array(transforms.depth_to_point_cloud(my_camera.intrinsic_matrix, obj_depth))

        if cam_pts.shape == (0,):
            print("No points are present in segmented point cloud. Please check your code. Continuing ...")
            continue

        world_pts = np.zeros((0,3))
        rt = camera.cam_view2pose(view_matrix)
        world_pts = transforms.transform_point3s(rt, cam_pts)

        world_pts_sample = world_pts[np.random.choice(range(world_pts.shape[0]), num_sample_pts), :]
        gt_pt_cloud = icp.mesh2pts(object_meshes[obj_index], len(world_pts_sample))

        transform = None  
        transformed = None 
        transform, transformed = icp.align_pts(gt_pt_cloud, world_pts_sample, 5000, 1e-50)

        position = None  
        grasp_angle = None
        position = transform[:-1,-1]
        grasp_angle = np.arctan2(transform[1,0], transform[0,0])

        # visualize grasp position using a big red sphere
        markers.append(sim.SphereMarker(position, radius = 0.02))

        # attempt grasping
        grasp_success = env.execute_grasp(position, grasp_angle)
        print(f"Grasp success: {grasp_success}")

        if grasp_success:  
            is_grasped[obj_index] = True

            path_conf = rrt(env.robot_home_joint_config,
                            env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.5, env)
            if path_conf is None:
                print("no collision-free path is found within the time budget. continuing ...")
            else:
                env.set_joint_positions(env.robot_home_joint_config)
                execute_path(path_conf, env)
        del markers
        p.removeAllUserDebugItems()
        env.robot_go_home()
        # env.reset_objects()
