#!/bin/bash

# Activate the conda environment
source /home/ttsai/miniconda3/bin/activate pytorch

# Training the model
python training.py --data_path '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset_divided_expand_combined' --val_path '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Validation_dataset' --log_file_path '/home/ttsai/Drone_contest_2/Log_file/training.log' --epochs 50 --batch_size 12 --save_path '/home/ttsai/Drone_contest_2/Save_path_all_test'

# Evaluate the model
python evaluation.py --model_path '/home/ttsai/Drone_contest_2/Save_path_all_test/best_model.pth' --test_data_path '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Validation_dataset' --log_file_path '/home/ttsai/Drone_contest_2/Log_file/evaluation.log' --batch_size 4

# Test the model
python testing.py --test_root '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Testing_dataset' --model_path '/home/ttsai/Drone_contest_2/Save_path_all_test/best_model.pth' --save_path '/home/ttsai/Drone_contest_2/Results_test/Results'

# Validate the model
python validation.py --mask_root '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Testing_dataset/labels' --pred_root '/home/ttsai/Drone_contest_2/Results_test/Results'
