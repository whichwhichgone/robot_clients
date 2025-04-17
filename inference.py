import sys, os
import numpy as np
import time, rospy, cv2, json
from typing import List, Tuple, Dict, Union, Any
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
import draccus
import requests
from scipy.spatial.transform import Rotation as R


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
    server_url: str = "http://172.16.78.10:35189/predict"
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

    # language instructions
    language_instructions = ["put the yellow corn into the red bowl.",]


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

    def env_update(self):
        scene_image: np.ndarray = self.scene_image_subscriber.get_current_image()
        wrist_image: np.ndarray = self.wrist_image_subscriber.get_current_image()
        rgb_image: np.ndarray = self.depth_image_subscriber.get_current_rgb_image()
        depth_img: np.ndarray = self.depth_image_subscriber.get_current_depth_image()
        robot_state: np.ndarray = self.robot.get_state()
        env_obs = {
            "depth": depth_img,                                         # shape (540, 960),    np.float32
            "rgb": rgb_image,                                           # shape (540, 960, 3), np.uint8
            "scene": scene_image,                                   # shape (480, 640, 3), np.uint8
            "wrist": wrist_image,                                   # shape (480, 640, 3), np.unit8
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

    def step(self, obs, language_instruction):
        img_scene = obs["img_scene"]
        img_hand_left = obs["img_hand_left"]
        img_hand_right = obs["img_hand_right"]

        # convert to bytes for sending
        img_scene_data = img_scene.tobytes()
        img_hand_left_data = img_hand_left.tobytes()
        img_hand_right_data = img_hand_right.tobytes()

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
        payload = {"instruction": language_instruction, "state": state.tolist()}
        files = {
            "json": json.dumps(payload),
            "img_scene": ("img_scene.txt", img_scene_data, "text/plain"),
            "img_hand_left": ("img_hand_left.txt", img_hand_left_data, "text/plain"),
            "img_hand_right": ("img_hand_right.txt", img_hand_right_data, "text/plain"),
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
            time.sleep(0.5)
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

        img_scene = Image.fromarray(img_scene)
        resized_scene = img_scene.resize(target_img_size)
        img_hand_left = Image.fromarray(img_hand_left)
        resized_hand_left = img_hand_left.resize(target_img_size)
        img_hand_right = Image.fromarray(img_hand_right)
        resized_hand_right = img_hand_right.resize(target_img_size)
        
        env_obs = {
            "img_scene": np.array(resized_scene),
            "img_hand_left": np.array(resized_hand_left),
            "img_hand_right": np.array(resized_hand_right),
        }
        return env_obs

    def roll_out(self, instruction, roll_out_len=200):
        for idx in range(roll_out_len):
            env_obs = self.env_update()
            env_obs = self.get_input_obs(env_obs)
            action = self.step(env_obs, instruction)
            if isinstance(action, list):
                for step_action in action:
                    
                    # this line is only for relative action, when you get absolute action, comment this line                    
                    step_action = self.rel2abs(step_action, self.robot.get_state())
                    self.robot.send_action(step_action)
                    time.sleep(0.1)
            elif isinstance(action, np.ndarray):
                self.robot.send_action(action)
                time.sleep(0.1)
            else:
                raise TypeError("actions must be list or ndarray")
    
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


@draccus.wrap()
def infer(cfg: InferConfig):
    robot = ClientRobot(cfg)

    for instruction in cfg.language_instructions:
        print(f"Executing instruction: {instruction}")
        robot.reset()
        robot.roll_out(instruction)


if __name__ == "__main__":
    infer()
