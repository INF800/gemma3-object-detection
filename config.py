from dataclasses import dataclass

import torch


@dataclass
class Configuration:
    project_name: str = "run-forrest-run"
    
    dataset_id: str = "ariG23498/license-detection-paligemma"

    model_id: str = "google/gemma-3-4b-pt"
    checkpoint_id: str = "sergiopaniego/gemma-3-4b-pt-object-detection-aug"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16

    batch_size: int = 1
    learning_rate: float = 2e-05
    epochs = 1
