from dataclasses import dataclass
from typing import Literal, Optional

import torch.cuda
import os

from vgcn_bert.prepare_data import preprocess
from vgcn_bert import Config


class Runner:
    def __init__(self, config: Config):
        self.config = config

    def run(self):
        preprocess(self.config)

        if self.config.model == "VGCN_BERT":
            from vgcn_bert.train_vgcn_bert import train as vb_train
            vb_train(config=self.config)
        else:
            print(f"{self.config.model} is not implemented!")
            Exception()

