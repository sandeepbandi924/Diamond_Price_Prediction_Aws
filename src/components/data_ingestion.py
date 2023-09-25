import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_tranformation import DataTranformation


##Intialize the Data ingestion configuration

@dataclass
class DataIngestionConfig:
   train_data_path:str = os.path.join('artifacts','train.csv')
   test_data_path:str = os.path.join('artifacts','test.csv')
   raw_data_path: str = os.path.join('artifacts','raw.csv')

#create a class for data ingestion
class DataIngestion :
   def __init__(self):
      self.ingestion_config = DataIngestionConfig()

   
   def intiate_data_ingestion(self):
      logging.info('Data Ingestion Method Starts')
      try:
         df = pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))  #reading dataset
         logging.info('Dataset read as pandas Dataframe')
                                                          #saving the dataset
         os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
         df.to_csv(self.ingestion_config.raw_data_path, index=False)

         logging.info('Train Test Split')       #we are doing train test split and we are saving it
         train_set,test_set = train_test_split(df, test_size=0.30, random_state=30)
                                                
         train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
         test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

         logging.info('Data Injestion is Completed')

         #we need to return output train data and test data

         return (self.ingestion_config.train_data_path,
                 self.ingestion_config.test_data_path
               
            ) 

      except Exception as e:
         logging.info('Error has happend at Data Ingestion')
         raise CustomException(e,sys)
      

#Run the data ingestion

# if __name__ == '__main__':
#    obj = DataIngestion()
#    train_data, test_data = obj.intiate_data_ingestion() #it is returning two variables
#    data_trasformation = DataTranformation()
#    train_arr, test_arr,_ = data_trasformation.initiate_data_transformation(train_data,test_data)
 


