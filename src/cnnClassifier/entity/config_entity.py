from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    source_PATH: Path



@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    params_image_size: tuple
    params_learning_rate: float
    params_alpha: float
    params_beta: float
    params_classes: int



@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir : Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path




@dataclass
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: tuple




@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    params_image_size: tuple
    params_batch_size: int