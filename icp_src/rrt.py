import sim
import pybullet as p
import numpy as np
import pdb

MAX_ITERS = 10000
delta_q = 1
PI = np.pi

def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    """
    Draw a line between two positions corresponding to the input configurations
    :param q_1: configuration 1
    :param q_2: configuration 2
    :param env: environment
    :param color: color of the line, please leave unchanged.
    """
    # obtain position of first point
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 9)[0]
    # obtain position of second point
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 9)[0]
    # draw line between points
    p.addUserDebugLine(point_1, point_2, color, 1.0)


def getNodeFromKey(str_node):
    return np.frombuffer(str_node, dtype = np.float64).reshape((1,6))[0]


def getDist(q1, q2, ord = "l2"):
    assert len(q1) == len(q2), "getDist() vector length mismatch..."
    error = 0
    if ord == "wrap":
        for i in range(len(q1)):
            error += min(2*PI - abs(q1[i] - q2[i]), abs(q1[i] - q2[i]))
    elif ord == "l2":
        error = np.linalg.norm(q2-q1)
    else:
        error = np.linalg.norm(q2-q1, ord = 1)
    
    return error


def nearest(G, q_rand):
    near_node = None
    min_dist = np.inf
    
    for nk in G.keys():
        node = getNodeFromKey(nk)
        dist = getDist(node, q_rand)
        if dist < min_dist:
            near_node = node
            min_dist = dist 
    
    return near_node


def semiRandomSample(steer_goal_p, q_goal):
    choice = np.random.choice([0, 1], 1, p = [1-steer_goal_p, steer_goal_p])[0]
    if choice:
        return q_goal
    else:
        sample = np.random.uniform(-PI, PI, 6)
        return sample


def steer(q_near, q_rand, delta_q):
    dist = getDist(q_near, q_rand)
    if dist < delta_q:
        return q_rand 
    else:
        step = delta_q*((q_rand - q_near)/dist)
        return q_near + step
    

def obstacleFree(q_new, env):
    return env.check_collision(q_new)

def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env):
    G = {}
    G[q_init.tostring()] = [q_init]
    
    # modified parameters
    steer_goal_p = 0.75
    delta_q = 0.3

    for _ in range(MAX_ITERS):
        q_rand = semiRandomSample(steer_goal_p, q_goal)
        q_near = nearest(G, q_rand)
        q_new = steer(q_near, q_rand, delta_q)
        dist = getDist(q_new, q_goal)
        
        # obstacleFree 
        if not env.check_collision(q_new):
            E = G[q_near.tostring()].copy()
            E.append(q_new)
            G[q_new.tostring()] = E
            
            if dist < delta_q:
                E.append(q_goal)
                return E
            else:
                visualize_path(q_near, q_new, env)
    
    return None



def execute_path(path_conf, env):
    markers = []
    for state in path_conf:
        curr_pos = p.getLinkState(env.robot_body_id, 9)[0]
        env.move_joints(state, speed = 0.05)
        markers.append(sim.SphereMarker(curr_pos))
    
    env.open_gripper()
    env.close_gripper()

    path_conf.reverse()
    for state in path_conf:
        env.move_joints(state, speed = 0.1)
    
    return None
