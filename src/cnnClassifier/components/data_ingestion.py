import os
import urllib.request as request
import zipfile
import mmap
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.config=config

    def download_file(self):

        if not os.path.exists(self.config.local_data_file):
            filename, headers= request.url.retrieve(
            url=self.config.source_URl,
            filname= self.config.local_data_file
            )
            logger.info(f"{filename} download! with the following info: \n{headers}")

        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
    
                      
    """def download_s3_file(self):
          
          #Note to use the s3 sdk  install the python 3.7 version for compatibility purpose
          
          if not os.path.exists(self.config.local_data_file):
              s3= boto3.client('s3')
              s3.download_file(
                  self.config.s3_bucket_name, 
                  self.config.s3_file_key,
                  self.config.local_data_file)
              logger.info(f"File {self.config.file_key} downloaded successfully from S3 bucket{self.config.bucket_name}!")
          else:
              logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")"""   

    def copy_file(self):

        if not os.path.exists(self.config.local_data_file):
            with open(self.config.source_PATH, 'rb') as src, open(self.config.local_data_file, 'wb') as dst:
                src_mmap = mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ)
                dst.write(src_mmap)
                src_mmap.close()

            logger.info(f"File copied from {self.config.source_PATH} to {self.config.local_data_file}")


    def extract_zip_files(self):

        """
        zip_file_path:str
        Extract the zip file into the data directory
        function returns None
        """
        unzip_path=self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)