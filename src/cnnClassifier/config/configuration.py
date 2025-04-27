from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
import ast 



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath= PARAMS_FILE_PATH):

        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config=self.config.data_ingestion

        create_directories([config.root_dir])
        data_ingestion_config= DataIngestionConfig(
          root_dir=config.root_dir,
          source_URL= config.source_URL,
          source_PATH= config.source_PATH,
          local_data_file=config.local_data_file,
          unzip_dir= config.unzip_dir,
          #s3_bucket_name=getattr(config, "s3_bucket_name", None),
          #s3_file_key=getattr(config, "s3_file_key", None)

    )

        return data_ingestion_config


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:

        config= self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config= PrepareBaseModelConfig(
            root_dir= Path(config.root_dir),
            base_model_path= Path(config.base_model_path),
            params_image_size= ast.literal_eval(self.params.IMAGE_SIZE),
            params_classes= self.params.CLASSES,
            params_alpha= self.params.alpha,
            params_beta=self.params.beta,
            params_learning_rate= self.params.LEARNING_RATE
        )

        return  prepare_base_model_config