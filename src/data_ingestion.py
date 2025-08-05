import os
import shutil
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.source_dir = os.path.join(RAW_DIR, "archive")
        self.file_names = self.config["bucket_file_names"]  # reuse same list from YAML

        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info("Data Ingestion Started (Local Mode)...")

    def copy_from_local(self):
        try:
            for file_name in self.file_names:
                src_path = os.path.join(self.source_dir, file_name)
                dest_path = os.path.join(RAW_DIR, file_name)

                if not os.path.exists(src_path):
                    raise FileNotFoundError(f"{file_name} not found in {self.source_dir}")

                # Special handling for large file
                if file_name == "animelist.csv":
                    df = pd.read_csv(src_path, nrows=5000000)
                    df.to_csv(dest_path, index=False)
                    logger.info("Copied animelist.csv with 5M rows")
                else:
                    shutil.copy(src_path, dest_path)
                    logger.info(f"Copied file: {file_name}")

        except Exception as e:
            logger.error("Error while copying data from local archive")
            raise CustomException("Failed to copy data", e)

    def run(self):
        try:
            logger.info("Starting Local Data Ingestion Process....")
            self.copy_from_local()
            logger.info("Local Data Ingestion Completed...")
        except CustomException as ce:
            logger.error(f"CustomException : {str(ce)}")
        finally:
            logger.info("Data Ingestion DONE...")


if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
