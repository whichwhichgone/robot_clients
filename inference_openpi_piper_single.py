import numpy as np
import time, json, cv2, os, sys, requests
from typing import List, Tuple, Dict, Union, Any
from PIL import Image, ImageOps
from pathlib import Path
from dataclasses import dataclass, field
import draccus

# Add path for the sdk packages, need to be specified according to your code
sys.path.append(
    "/home/xhjqr/gello_ws/agilex_agent_env_py/gello_arm_python/wl_ros2_ws/src/wl_robot_python_sdk/wl_robot_python_sdk"
)

# NOTE: Need to initialize the env with init_env_piper.sh before running this script 
from motor_controller import ArmController
from utils.high_freq_timer import HighFreqTimer
from utils.cameras import ThreadedVideoCapture


@dataclass
class InferConfig:
    server_url: str = "http://172.16.78.10:32739/predict"
    debug_mode: bool = True
    debug_dir: str = Path("./imgs_debug")

    # initial position for the robot
    init_action: list = field(
        default_factory=lambda: [
            0.02147573103039857,
            -0.19021361769781908,
            -0.21629129109187506,
            -0.003067961575771161,
            0.9802137234589248,
            0.05829126993965428,
            1.0,
            0.02147573103039857,
            -0.19021361769781908,
            -0.21629129109187506,
            -0.003067961575771161,
            0.9802137234589248,
            0.05829126993965428,
            1.0,
        ]
    )

    # language instructions
    language_instructions = [
        "pick up the left corn and put it into the red bowl",
        #"arrange the blocks on the table into an L-shape, first put blue at bottom, second put red at top, last put yellow at right",
    ]


class ClientRobot:
    def __init__(self, cfg: InferConfig):
        self.scene_image_subscriber = ThreadedVideoCapture("front_head")
        self.left_image_subscriber = ThreadedVideoCapture("left_hand")
        self.right_image_subscriber = ThreadedVideoCapture("right_hand")

        self.robot = ArmController(protect_threshold=0.8)
        self.protection_triggered = False
        self.reset(np.array(cfg.init_action))

        self.action = np.array(cfg.init_action)
        self.timer = HighFreqTimer(0.01, lambda: self.callback_action(self.action))
        self.timer.start()

        self.server_url = cfg.server_url
        self.debug = cfg.debug_mode
        self.debug_dir = cfg.debug_dir

    def callback_action(self, action):
        if self.protection_triggered:
            print("[CALLBACK INFO 1] Protection triggered, control not issued.")
            return

        ok = self.robot.publish(
            left_joints=action[:6],
            left_gripper=action[6],
            right_joints=action[7:13],
            right_gripper=action[13],
            do_safety_check=True,
        )

        if not ok:
            print("[CALLBACK INFO 2] Protection triggered, control not issued.")
            self.protection_triggered = True
            return

    def send_action(self, action):
        # maintain the self.action for timer callback
        self.action = action

    def env_update(self):
        scene_success, scene_image = self.scene_image_subscriber.read()
        left_success, left_image = self.left_image_subscriber.read()
        right_success, right_image = self.right_image_subscriber.read()

        assert scene_success and left_success and right_success, (
            f"Failed to read from one of the cameras: "
            f"scene_success={scene_success}, "
            f"left_success={left_success}, "
            f"right_success={right_success}"
        )
        
        # rotate the image to the correct orientation
        left_image = cv2.rotate(left_image, cv2.ROTATE_180)
        right_image = cv2.rotate(right_image, cv2.ROTATE_180)

        # convert BGR to RGB
        scene_image = cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB)
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

        robot_state: np.ndarray = self.robot.get_current_positions()
        env_obs = {
            "scene": scene_image,  # shape (240, 424, 3), np.uint8
            "left": left_image,    # shape (240, 320, 3), np.uint8
            "right": right_image,  # shape (240, 320, 3), np.uint8
            "state": robot_state,  # shape (14,)          np.float64
        }

        if self.debug:
            scene_image = Image.fromarray(scene_image)
            left_image = Image.fromarray(left_image)
            right_image = Image.fromarray(right_image)
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            scene_image.save(self.debug_dir / "raw_scene.png")
            left_image.save(self.debug_dir / "raw_left.png")
            right_image.save(self.debug_dir / "raw_right.png")
        return env_obs

    def reset(self, init_action):
        time.sleep(5)
        current_joint = self.robot.get_current_positions()
        steps = 100
        for i in range(1, steps + 1):
            if self.protection_triggered:
                print(
                    "[INFO] Protection triggered, interpolation initialization aborted"
                )
                break
            ratio = float(i) / float(steps)
            cmd = current_joint + ratio * (init_action - current_joint)
            ok = self.robot.publish(
                left_joints=cmd[:6],
                left_gripper=cmd[6],
                right_joints=cmd[7:13],
                right_gripper=cmd[13],
                do_safety_check=True,
            )

            if not ok:
                print("[INFO] Protection triggered, initialization control not issued")
                self.protection_triggered = True
                break
            time.sleep(0.05)

    def step(self, obs, language_instruction):
        img_scene = obs["img_scene"]
        img_hand_left = obs["img_hand_left"]
        img_hand_right = obs["img_hand_right"]
        state = obs["state"]

        # convert to bytes for sending
        img_scene_data = img_scene.tobytes()
        img_hand_left_data = img_hand_left.tobytes()
        img_hand_right_data = img_hand_right.tobytes()

        if self.debug:
            img_scene = Image.fromarray(img_scene)
            img_hand_left = Image.fromarray(img_hand_left)
            img_hand_right = Image.fromarray(img_hand_right)
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            img_scene.save(self.debug_dir / "scene.png")
            img_hand_left.save(self.debug_dir / "hand_left.png")
            img_hand_right.save(self.debug_dir / "hand_right.png")

        # compose the request payload
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

        resp_action = np.array(action)
        action = np.array_split(resp_action, 16)
        #action = action[:20]
        return action

    def get_input_obs(self, env_obs, target_img_size=(224, 224)):
        # covert the original obs to the target format for the model.
        img_scene = env_obs["scene"]
        img_hand_left = env_obs["left"]
        img_hand_right = env_obs["right"]
        state = env_obs["state"]

        img_scene = Image.fromarray(img_scene)
        width, height = img_scene.size
        max_side = max(width, height)
        # resized_scene = ImageOps.pad(img_scene, (max_side, max_side), method=Image.LANCZOS, color=(0, 0, 0))
        resized_scene = img_scene.resize(target_img_size, Image.BICUBIC)

        img_hand_left = Image.fromarray(img_hand_left)
        width, height = img_hand_left.size
        max_side = max(width, height)
        # resized_hand_left = ImageOps.pad(img_hand_left, (max_side, max_side), method=Image.LANCZOS, color=(0, 0, 0))
        resized_hand_left = img_hand_left.resize(target_img_size, Image.BICUBIC)

        img_hand_right = Image.fromarray(img_hand_right)
        width, height = img_hand_right.size
        max_side = max(width, height)
        # resized_hand_right = ImageOps.pad(img_hand_right, (max_side, max_side), method=Image.LANCZOS, color=(0, 0, 0))
        resized_hand_right = img_hand_right.resize(target_img_size, Image.BICUBIC)

        env_obs = {
            "img_scene": np.array(resized_scene),
            "img_hand_left": np.array(resized_hand_left),
            "img_hand_right": np.array(resized_hand_right),
            "state": state,
        }
        return env_obs

    def roll_out(self, instruction, roll_out_len=200):
        for idx in range(roll_out_len):
            env_obs = self.env_update()
            env_obs = self.get_input_obs(env_obs)
            action = self.step(env_obs, instruction)
            left_action = np.array(
                [0.02147573103039857,
                -0.19021361769781908,
                -0.21629129109187506,
                -0.003067961575771161,
                0.9802137234589248,
                0.05829126993965428,
                1.0])
            if isinstance(action, list):
                for step_action in action:
                    time.sleep(0.05)
                    step_action[6] -= 0.1
                    step_action = np.concatenate([left_action, step_action], axis=0)
                    self.send_action(step_action)
            else:
                raise TypeError("actions must be list of ndarray, each with shape (14,)")

    def roll_out_test(self):
        ### this func just for dummy test
        with open("/home/xhjqr/zhaowei/data/episode_60/data.json", "r") as f:
            action = json.load(f)
        action = [elem["joint"] for elem in action]

        cnt = 0
        for step_action in action:
            self.send_action(step_action)
            time.sleep(0.05)
            cnt += 1
            if cnt % 20 == 0:
                time.sleep(2)


@draccus.wrap()
def infer(cfg: InferConfig):
    robot = ClientRobot(cfg)

    for instruction in cfg.language_instructions:
        print(f"Executing instruction: {instruction}")
        robot.roll_out(instruction)
        #robot.roll_out_test()


if __name__ == "__main__":
    infer()
