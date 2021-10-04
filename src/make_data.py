import os
import sys

from omegaconf import DictConfig, OmegaConf
import hydra

from prepro.data_builder import format_df_to_bert
from others.logging import logger, init_logger


@hydra.main(config_path="../conf/make_data", config_name="config")
def main(cfg: DictConfig) -> None:
    init_logger(cfg.dirs.log_file)

    modes = ["df_to_bert"]
    if cfg.mode not in modes:
        logger.error(f"Incorrect mode. Please choose one of {modes}")
        sys.exit("Stop")

    # print(OmegaConf.to_yaml(cfg))
    print(os.getcwd())

    # Make bert input file for train and valid from df file
    if cfg.mode in ["df_to_bert"]:
        format_df_to_bert(cfg)


if __name__ == "__main__":
    main()
