import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.impute import SimpleImputer #handling missing values
from sklearn.preprocessing import StandardScaler #handling feature scaling
from sklearn.preprocessing import OrdinalEncoder # oridinal encoding
##pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
   #pkl file -> model when ever we create we try to save in the harddisk --> it will converted into pkl file a serialized file
   preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTranformation:
   def __init__(self):
      self.data_transformation_cofig = DataTransformationConfig()

   def get_data_transformation_object(self):
      try:
         logging.info('Data Transformation Intiated')
         #Define which col should be ordinal- encoded and which should be scaled
         categorical_cols=['cut', 'color', 'clarity']
         numerical_cols = ['carat', 'depth','table','x','y','z']

         #Define the custom ranking for each ordinal variable
         cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
         color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
         clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

         logging.info('Pipeline Intiated')

         ##Numerical pipeline
         num_pipeline = Pipeline(
            steps=[
               ('impute', SimpleImputer(strategy='median')),
               ('scaler',StandardScaler())

            ]

         )

         ##Categorical pipeline
         cat_pipeline = Pipeline(
            
            steps=[
               ('impute', SimpleImputer(strategy='most_frequent')),
               ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
               ('scaler', StandardScaler())

            ]

         )

         preprocessor = ColumnTransformer([

            ('num_pipeline', num_pipeline , numerical_cols),
            ('cat_pipeline', cat_pipeline,categorical_cols)
         ])

         return preprocessor
      
         logging.info('Pipeline Completed')


      except Exception as e:
         logging.info('Error in Data Transformation')
         raise CustomException(e,sys)
      
   
   def initiate_data_transformation(self,train_path,test_path):
      try:
         #reading train and test data
         train_df = pd.read_csv(train_path)
         test_df = pd.read_csv(test_path)

         logging.info('Reading train and test data completed')
         logging.info(f'Train DataFrame Head: \n {train_df.head().to_string()}')
         logging.info(f'Test DataFrame Head: \n {test_df.head().to_string()}')

         logging.info('Obtaining preprocessor object')

         preprocessor_obj = self.get_data_transformation_object()


         target_col_name = 'price'
         drop_column = [target_col_name , 'id']

         input_feature_train_df = train_df.drop(columns=drop_column, axis=1)
         target_feature_train_df = train_df[target_col_name]

         input_feature_test_df = test_df.drop(columns=drop_column, axis=1)
         target_feature_test_df = test_df[target_col_name]

         #Transforming using preprocessor obj
         input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
         input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

         logging.info('Applying preprocessing on training and testing datasets.')


         train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
         test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


         save_object(
            file_path=self.data_transformation_cofig.preprocessor_obj_file_path,
            obj = preprocessor_obj
         )
         logging.info('Preprocessing pickle file saved')

         return(
            train_arr,
            test_arr,
            self.data_transformation_cofig.preprocessor_obj_file_path
         )
      except Exception as e:
         logging.info('Error in initiate Data Transformation')
         raise CustomException(e,sys)




