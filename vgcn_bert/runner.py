from dataclasses import dataclass
from typing import Literal, Optional

import torch.cuda
import os


@dataclass()
class Config:
    needs_preprocess: bool = True
    dataset_format: Literal["sst", "cola", "auto"] = "sst"
    dataset_path: str = "data/SST-2"
    delete_stopwords: bool = True
    higher_threshold_for_sw: bool = False
    use_larger_cw: bool = False
    dumps_objects: bool = False
    dump_path: Optional[str] = None

    tf_idf_mode: Literal["only_tf", "all_tfidf", "all_tf_train_valid_idf"] = "only_tf"
    bert_model_for_preprocess: Optional[str] = "bert-base-uncased"
    bert_tokenizer_lower: bool = True

    random_seed = 42

    disable_cuda: bool = False

    def __post_init__(self):
        self.device = 'cuda' if torch.cuda.is_available() and not self.disable_cuda else 'cpu'
        self.dataset_name = os.path.basename(self.dataset_path)
        if self.dumps_objects and self.dump_path is None:
            self.dump_path = os.path.join(
                os.path.dirname(self.dataset_path),
                "preprocess",
                self.dataset_name
            )
        if self.dataset_format == "auto":
            print("'auto' format processor hasn't implemented!")
            exit(1)

    def __repr__(self):
        print(f"""
Config(
    {self.needs_preprocess=},
    {self.dataset_name}
    {self.dataset_format=},
    {self.dataset_path=},
    {self.delete_stopwords=},
    {self.higher_threshold_for_sw},
    {self.use_larger_cw=} (suggested for "mr", "sst", "cola"|capture almost whole sentence),
    {self.dumps_objects=},
    {self.dump_path=},
    
    {self.tf_idf_mode=} ('only_tf' or 'all_tfidf' or 'all_tf_train_valid_idf'),
    {self.bert_model_for_preprocess=} (only the tokenizer would be used),
    bert_tokenizer_lowercasing={self.bert_tokenizer_lower},
    
    {self.random_seed=},
    
    {self.device=},
    
""")
