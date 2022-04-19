import os
import pathlib


class Config:
    def __init__(self, model_name):

        self.model_name = model_name

        ##########################################
        ############ Model Parameters ############
        ##########################################

        self.num_filters = 196
        self.num_classes = 4*5000

        ##########################################
        ############# Hyperparameters ############
        ##########################################

        self.learning_rate = 0.01
        self.num_epochs = 2
        self.batch_size = 2

        ##########################################
        ############ Input Directories ###########
        ##########################################

        self.data_dir = '../../data/cnn/human/'

        ##########################################
        ############       Data       ############
        ##########################################

        self.train_chr = [1,3,5,7,9,11]
        self.valid_chr = [2,4]
        self.test_chr = [22]

        ##########################################
        ############ Output Locations ############
        ##########################################

        self.model_dir = '../../data/cnn/models/'
        self.output_directory = '../../data/cnn/outputs/%s/' % model_name

        for file_path in [self.model_dir, self.output_directory]:
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                print("Creating directory %s" % file_path)
                pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
            else:
                print("Directory %s exists" % file_path)
