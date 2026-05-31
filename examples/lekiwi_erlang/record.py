# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path

from lerobot.common.control_utils import init_keyboard_listener
from lerobot.datasets import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi_erlang import LeKiwiErlangClient, LeKiwiErlangClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

NUM_EPISODES = 15
FPS = 20
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 60
TASK_DESCRIPTION = "pick up the charger then pick up the wire into the box"
HF_REPO_ID = "<hf_username>/<dataset_repo_id>"
DATASET_REPO_ID = "local/my_dataset"
DATASET_ROOT = "/home/starz/桌面/tidy_up_desk"


def main(resume: bool = False):
    # Create the robot and teleoperator configurations
    robot_config = LeKiwiErlangClientConfig(remote_ip="172.18.100.100", id="my_lekiwi")
    leader_arm_config = SO101LeaderConfig(port="/dev/ttyACM0", id="my_lekiwi_leader")
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    # Initialize the robot and teleoperator
    robot = LeKiwiErlangClient(robot_config)
    leader_arm = SO101Leader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # Create or resume the dataset
    dataset_root = Path(DATASET_ROOT)
    if resume and dataset_root.exists():
        log_say(f"Resuming existing dataset from {DATASET_ROOT}")
        dataset = LeRobotDataset.resume(
            repo_id=DATASET_REPO_ID,
            root=DATASET_ROOT,
            image_writer_threads=4,
        )
    else:
        log_say(f"Creating new dataset at {DATASET_ROOT}")
        dataset = LeRobotDataset.create(
            repo_id=DATASET_REPO_ID,
            root=DATASET_ROOT,
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )

    # Connect the robot and teleoperator
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    # Initialize the keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_record")

    try:
        if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
            raise ValueError("Robot or teleop is not connected!")

        teleop_action_processor, robot_action_processor, robot_observation_processor = (
            make_default_processors()
        )

        print("Starting record loop...")
        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {recorded_episodes}")

            # Main record loop
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset,
                teleop=[leader_arm, keyboard],
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (
                (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=[leader_arm, keyboard],
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save episode
            dataset.save_episode()
            recorded_episodes += 1
    finally:
        # Clean up
        log_say("Stop recording")
        robot.disconnect()
        leader_arm.disconnect()
        keyboard.disconnect()
        listener.stop()

        dataset.finalize()
        # dataset.push_to_hub() # 注释或删除这行以禁止上传到 Hugging Face


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record demonstrations for LeRobot LeKiwi robot")
    parser.add_argument(
        "--resume",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        help="Resume recording to an existing dataset (default: False)",
    )
    args = parser.parse_args()
    main(resume=args.resume)
