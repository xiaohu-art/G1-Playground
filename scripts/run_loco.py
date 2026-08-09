import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from g1_playground.dds import create_dds_topic_to_communicate_with_g1

from g1_playground.policies.loco.policy import LocoPolicy
from g1_playground.robot import UnitreeG1Robot


@hydra.main(version_base=None, config_path="../configs", config_name="run_loco")
def main(cfg: DictConfig) -> None:
    policy = LocoPolicy(OmegaConf.to_container(cfg.policy, resolve=True))

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    print(f"DDS: channel {cfg.dds.channel_id}, interface {cfg.dds.network_interface}")
    try:
        input("Press Enter to continue...")
    except KeyboardInterrupt:
        print("\nAborted before start.")
        return

    create_dds_topic_to_communicate_with_g1(cfg.dds.channel_id, cfg.dds.network_interface)

    robot = UnitreeG1Robot(policy=policy)

    try:
        robot.initialize()
        robot.start()
        print("Running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        robot.stop()
        print("Robot stopped. Exiting.")


if __name__ == "__main__":
    main()
    sys.exit(0)
