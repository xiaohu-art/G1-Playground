import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import time

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="g1", help="Configuration name")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg: RlPipelineCfg = ConfigManager(config_name=args.config).get_cfg()
    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, cfg.pipeline_type)
    logger.info(f"Using pipeline: {cfg.pipeline_type} -> {pipeline_class}")
    pipeline = pipeline_class(cfg=cfg)

    if not cfg.env.is_sim:
        pipeline.prepare()

    try:
        while pipeline.running:
            time_start = time.time()
            pipeline.step()
            time_diff = pipeline.dt - (time.time() - time_start)
            if cfg.run_fullspeed:
                continue
            if time_diff > 0:
                time.sleep(time_diff)
            elif not cfg.env.is_sim:
                logger.error(f"Warning: frame drop -> {time_diff}")
                if time_diff < -0.2:
                    logger.critical("Exiting due to excessive frame drop")
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted by operator")
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
