import logging
import shutil
from ludwig.api import LudwigModel

# clean out prior results
shutil.rmtree('./results', ignore_errors=True)

# Define Ludwig model object that drive model training
model = LudwigModel(config='./config.yaml', logging_level=logging.INFO)

# initiate model training
train_stats, _, output_directory = model.train(dataset='./data/nlu.csv', 
                                                experiment_name='experiment', 
                                                model_name='nlu')