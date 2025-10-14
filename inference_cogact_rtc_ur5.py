import sys, os
import numpy as np
import time, rospy, cv2, json
from typing import List, Tuple, Dict, Union, Any
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
import draccus
import requests
import threading
from scipy.spatial.transform import Rotation as R

from collections import deque


### Add path for the required packages
sys.path.append("/home/robot/UR_Robot_Arm_Show/tele_ws/src/tele_ctrl_jeff/scripts")
sys.path.append("/home/robot/UR_Robot_Arm_Show/")
sys.path.append("/home/robot/UR_Robot_Arm_Show/utils/")

### Subscribe the sensors
from depth_camera_v2 import DepthCameraSubscriber
from wrist_camera import WristSubscriber
from scene_camera import SceneSubscriber

### Subscribe the robot 
from arm_robot import ArmRobot


@dataclass
class InferConfig:
    server_url: str = "http://172.16.78.10:32255/predict"
    debug_mode: bool = True
    debug_dir: str = Path("/home/robot/workspace/ManiStation/UR5_cogact/imgs_debug")

    # initial absolute position
    init_action: list = field(
        default_factory=lambda: 
        [
            -0.16917512103213775,
            -0.31559707628103384,
            0.19853144723339541,
            -0.44689941080662177,
            -0.8941862128013452,
            0.02208828288125548,
            0.014968006414996177,
            0.
        ]
    )
    action_exec_s: int = 8
    action_chunk_length: int = 16

    # language instructions
    language_instructions = ["pick up the corn and put it into the blue bowl",]


class ClientRobot:

    def __init__(self, cfg: InferConfig):
        self.scene_image_subscriber = SceneSubscriber()
        self.depth_image_subscriber = DepthCameraSubscriber()
        self.wrist_image_subscriber = WristSubscriber()
        rospy.init_node("inference_node")
        self.robot = ArmRobot()
        self.server_url = cfg.server_url
        self.debug = cfg.debug_mode
        self.debug_dir = cfg.debug_dir
        self.init_action = cfg.init_action

        # real time action chunking, global shared state
        self.init_flag = True
        self.time_step = 0
        self.act_chunk_cur = None
        self.obs_cur = None
        self.mutex = threading.Lock()
        self.condition = threading.Condition(self.mutex)
        self.s_min = cfg.action_exec_s
        self.action_chunk_length = cfg.action_chunk_length

    def env_update(self):
        scene_image: np.ndarray = self.scene_image_subscriber.get_current_image()
        wrist_image: np.ndarray = self.wrist_image_subscriber.get_current_image()
        rgb_image: np.ndarray = self.depth_image_subscriber.get_current_rgb_image()
        depth_img: np.ndarray = self.depth_image_subscriber.get_current_depth_image()
        robot_state: np.ndarray = self.robot.get_state()
        env_obs = {
            "depth": depth_img,                                         # shape (540, 960),    np.float32
            "rgb": rgb_image,                                           # shape (540, 960, 3), np.uint8
            "scene": scene_image,                                       # shape (480, 640, 3), np.uint8
            "wrist": wrist_image,                                       # shape (480, 640, 3), np.unit8
            "state": robot_state,                                       # shape (8,)           np.float64
        }

        if self.debug:
            scene_image = Image.fromarray(scene_image)
            wrist_image = Image.fromarray(wrist_image)
            rgb_image = Image.fromarray(rgb_image)
            depth_img = Image.fromarray((depth_img * 255).astype(np.uint8))

            scene_image.save(self.debug_dir / "raw_scene.png")
            wrist_image.save(self.debug_dir / "raw_wrist.png")
            rgb_image.save(self.debug_dir / "raw_rgb.png")
            depth_img.save(self.debug_dir / "raw_depth.png")
        return env_obs

    def reset(self):
        self.robot.init_action(self.init_action)

    def step_init(self, obs, language_instruction):
        img_scene = obs["img_scene"]
        img_hand_left = obs["img_hand_left"]
        img_hand_right = obs["img_hand_right"]
        depth = obs["depth"]

        # convert to bytes for sending
        img_scene_data = img_scene.tobytes()
        img_hand_left_data = img_hand_left.tobytes()
        img_hand_right_data = img_hand_right.tobytes()
        depth_data = depth.astype(np.float32).tobytes()

        if self.debug:
            img_scene = Image.fromarray(img_scene)
            img_hand_left = Image.fromarray(img_hand_left)
            img_hand_right = Image.fromarray(img_hand_right)
            img_scene.save(self.debug_dir / "scene.png")
            img_hand_left.save(self.debug_dir / "hand_left.png")
            img_hand_right.save(self.debug_dir / "hand_right.png")
        
        # compose the request payload
        # self.robot.get_state()
        state = self.robot.get_state()
        payload = {
            "instruction": language_instruction,
            "action_part_prev": None,
            "state": state.tolist(),
            "delay": None,
            "s": None,
        }
        files = {
            "json": json.dumps(payload),
            "img_scene": ("img_scene.txt", img_scene_data, "text/plain"),
            "img_hand_left": ("img_hand_left.txt", img_hand_left_data, "text/plain"),
            "img_hand_right": ("img_hand_right.txt", img_hand_right_data, "text/plain"),
            "depth": ("depth.txt", depth_data, "text/plain"),
        }

        timeout_cnt = 0
        while True:
            try:
                action = requests.post(self.server_url, files=files)
                # if action.headers._store["server"][1] != "nginx" and action.status_code == 200:
                if action.status_code == 200:
                    action = action.json()
                    break
                else:
                    print("Error return code, retry")
            except requests.RequestException:
                print("Error request sending, retry")
            #time.sleep(0.5)
            timeout_cnt += 1
            if timeout_cnt >= 10:
                raise ValueError("Connection error, check the internet")

        action = [np.array(elem) for elem in action]
        return action


    def get_input_obs(self, env_obs, target_img_size=(224, 224)):
        # covert the original obs to the target format for model.
        img_scene = env_obs["scene"]
        img_hand_left = env_obs["wrist"]
        img_hand_right = env_obs["rgb"]
        depth = env_obs["depth"]

        img_scene = Image.fromarray(img_scene)
        resized_scene = img_scene.resize(target_img_size)
        img_hand_left = Image.fromarray(img_hand_left)
        resized_hand_left = img_hand_left.resize(target_img_size)
        img_hand_right = Image.fromarray(img_hand_right)
        resized_hand_right = img_hand_right.resize(target_img_size)
        depth = Image.fromarray(depth, mode="L")
        resized_depth = depth.resize(target_img_size)
        
        env_obs = {
            "img_scene": np.array(resized_scene),
            "img_hand_left": np.array(resized_hand_left),
            "img_hand_right": np.array(resized_hand_right),
            "depth": np.array(resized_depth),
        }
        return env_obs

    def roll_out(self, instruction, roll_out_len=200):
        for idx in range(roll_out_len):
            if self.init_flag:
                env_obs = self.get_input_obs(self.env_update())
                action = self.step_init(env_obs, instruction)

                # 0. init the global shared state
                self.action_chunk_cur = action
                self.obs_cur = env_obs
                self.time_step = 0
                self.init_flag = False

                # 1. create a background thread for the inference loop
                self.thread = threading.Thread(target=self.inference_loop, args=(instruction,))
                self.thread.start()
            
            # set the control frequency to about 20Hz
            time.sleep(0.01)
            env_obs = self.get_input_obs(self.env_update())
            action = self.get_action(env_obs)
            action = self.rel2abs(action, self.robot.get_state())
            self.robot.send_action(action)
        print("roll out done")


    def roll_out_test(self):
        ### this func just for dummy test
        action = np.load("/home/robot/workspace/ManiStation/UR5_cogact/pose_data.npy")
        action = [elem for elem in action]

        for step_action in action:
            #step_action = self.rel2abs(step_action, self.robot.get_state())
            self.robot.send_action(step_action)
            time.sleep(0.1)
    
    def rel2abs(self, rel_action, curr_state):
        assert curr_state.shape[0] == 8, "wrong format robot state"
        assert rel_action.shape[0] == 7, "wrong format relative action"
        xyz_before = curr_state[:3]
        quat_before = curr_state[3:7]
        gripper_before = curr_state[7]

        r = R.from_quat(quat_before)
        euler_before = r.as_euler('xyz', degrees=False)
        state_after = np.hstack((xyz_before, euler_before, gripper_before))
        state_after = state_after + rel_action

        xyz_after = state_after[:3]
        euler_after = state_after[3:6]

        r = R.from_euler('xyz', euler_after, degrees=False)
        quat_after = r.as_quat()

        if rel_action[-1:] > 0.4:
            abs_action = np.hstack((xyz_after, quat_after, np.array([1.0])))
        else:
            abs_action = np.hstack((xyz_after, quat_after, np.array([0.0])))
        return abs_action
    
    def get_action(self, obs_env):
        with self.condition:
            self.obs_cur = obs_env
            action = self.action_chunk_cur[self.time_step]
            self.time_step += 1
            self.condition.notify_all()
        return action

    def inference_loop(self, instruction):
        with self.condition:
            # Initialize with 4 actions for the first element
            queue_delay = deque([3], maxlen=10)
            while True:
                while self.time_step < self.s_min:
                    self.condition.wait()
                s = self.time_step
                action_part_prev = self.action_chunk_cur[s: ]
                obs = self.obs_cur
                delay = max(queue_delay)
                self.condition.release()
                print(f"Instruction: {instruction}, s: {s}, delay: {delay}, time_step: {self.time_step}")
                print(f"Delay queue: {queue_delay}")

                try:
                    action_chunk_new = self.guided_inference(instruction, obs, action_part_prev, delay, s)
                except Exception as e:
                    print(f"Error in guided_inference: {e}")
                finally:
                    self.condition.acquire()
                self.action_chunk_cur = action_chunk_new
                self.time_step = self.time_step - s
                queue_delay.append(self.time_step)

    def guided_inference(self, instruction, obs, action_part_prev, delay, s):
        img_scene = obs["img_scene"]
        img_hand_left = obs["img_hand_left"]
        img_hand_right = obs["img_hand_right"]
        depth = obs["depth"]

        # convert to bytes for sending
        img_scene_data = img_scene.tobytes()
        img_hand_left_data = img_hand_left.tobytes()
        img_hand_right_data = img_hand_right.tobytes()
        depth_data = depth.astype(np.float32).tobytes()

        if self.debug:
            img_scene = Image.fromarray(img_scene)
            img_hand_left = Image.fromarray(img_hand_left)
            img_hand_right = Image.fromarray(img_hand_right)
            img_scene.save(self.debug_dir / "scene.png")
            img_hand_left.save(self.debug_dir / "hand_left.png")
            img_hand_right.save(self.debug_dir / "hand_right.png")
        
        # compose the request payload
        action_part_prev = np.asarray(action_part_prev)
        action_part_prev = np.pad(action_part_prev, ((0, self.action_chunk_length - len(action_part_prev)), (0, 0)), mode='constant')
        state = self.robot.get_state()
        payload = {
                "instruction": instruction,
                "action_part_prev": action_part_prev.tolist(),
                "state": state.tolist(),
                "delay": str(delay),
                "s": str(s),
            }
        files = {
            "json": json.dumps(payload),
            "img_scene": ("img_scene.txt", img_scene_data, "text/plain"),
            "img_hand_left": ("img_hand_left.txt", img_hand_left_data, "text/plain"),
            "img_hand_right": ("img_hand_right.txt", img_hand_right_data, "text/plain"),
            "depth": ("depth.txt", depth_data, "text/plain"),
        }

        action = requests.post(self.server_url, files=files)
        if action.status_code == 200:
            action = action.json()
        else:
            raise ValueError("Failed to get valid response from server. Status code: " + str(action.status_code))
        action = [np.array(elem) for elem in action]
        return action


@draccus.wrap()
def infer(cfg: InferConfig):
    robot = ClientRobot(cfg)

    for instruction in cfg.language_instructions:
        print(f"Executing instruction: {instruction}")
        robot.reset()
        robot.roll_out(instruction)


if __name__ == "__main__":
    infer()
