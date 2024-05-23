import os
import zipfile
import boto3
from ImageClassifier import logger
from ImageClassifier.utils.common import get_size
from ImageClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            s3 = boto3.client("s3", aws_access_key_id= self.config.access_key, aws_secret_access_key= self.config.secret_key)
            logger.info("Downloading started")
            s3.download_file(
                Filename=self.config.local_data_file,
                Bucket=self.config.bucket_name,
                Key=self.config.folder_name   
            )
            logger.info(f"{self.config.local_data_file} downloaded!")
        else:
            logger.info(f"{self.config.local_data_file} already exists")
            # logger.info(f"Size of {self.config.local_data_file} is {get_size(self.config.local_data_file)}")
            
    
    def unzip_file(self):
        logger.info(f"{self.config.local_data_file} unzip started")
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
   