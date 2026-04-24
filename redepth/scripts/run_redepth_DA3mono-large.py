import sys
import os
from omegaconf import OmegaConf
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../model")))

from redepth.coach.DA3_monolarge_coach import DA3Trainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    trainer = DA3Trainer(config)
    trainer.train()
