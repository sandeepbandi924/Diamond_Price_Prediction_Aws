
import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_tranformation import DataTranformation
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src.utils import save_object,evaluate_model


if __name__ == '__main__':
   obj = DataIngestion()
   train_data, test_data = obj.intiate_data_ingestion() #it is returning two variables
   data_trasformation = DataTranformation()
   train_arr, test_arr,_ = data_trasformation.initiate_data_transformation(train_data,test_data)
   model_trainer =ModelTrainer()
   model_trainer.initiate_model_training(test_arr, test_arr)