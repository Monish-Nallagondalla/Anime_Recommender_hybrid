import os
import pandas as pd
import numpy as np
import joblib
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
import sys

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir

        self.rating_df = None
        self.anime_df = None
        self.X_train_array = None
        self.X_test_array = None
        self.y_train = None
        self.y_test = None

        self.user2user_encoded = {}
        self.user2user_decoded = {}
        self.anime2anime_encoded = {}
        self.anime2anime_decoded = {}

        os.makedirs(self.output_dir,exist_ok=True)
        logger.info("DataProcessing Intialized")
    

    def load_data(self,usecols):
        try:
            self.rating_df = pd.read_csv(self.input_file , low_memory=True,usecols=usecols)
            logger.info("Data loaded sucesfully for Data Processing")
        except Exception as e:
            raise CustomException("Failed to load data",sys)

    def filter_users(self,min_rating):
        try:
            n_ratings = self.rating_df["user_id"].value_counts()
            n_ratings = self.rating_df[self.rating_df['user_id'].isin(self.n_ratings[n_ratings>=400])].copy()

            logger.info("Filtered users successfully....")
        except Exception as e:
            raise CustomException("Failed to filter data",sys)
        

    def scale_rating(self):
        try:
            min_rating =min(self.rating_df["rating"])
            max_rating = max(self.rating_df['rating'])
            self.rating_df["rating"] = self.rating_df["rating"].apply(lambda x: (x-min_rating)/(max_rating-min_rating)).values.astype(np.float64)
            logger.info("Scalind done for Processing ")
        except Exception as e:
            raise CustomException("Failed to scale data",sys)
        

    def 