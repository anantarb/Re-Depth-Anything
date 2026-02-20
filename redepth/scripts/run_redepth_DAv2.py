import sys
import os
from omegaconf import OmegaConf
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from redepth.coach.DAv2_coach import DAv2Trainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    trainer = DAv2Trainer(config)
    trainer.train()
