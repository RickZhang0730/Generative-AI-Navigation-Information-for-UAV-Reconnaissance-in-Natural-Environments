# Generative-AI-Navigation-Information-for-UAV-Reconnaissance-in-Natural-Environments
A project focused on constructing autonomous UAV navigation datasets for natural environments, leveraging generative AI to enhance and generate training data from images of Taiwan's roads and rivers.

## Installation
To creating the environment, you can use the provided env.yml file to create a conda environment.
```bash
conda env create -f env.yml
conda activate your_environment_name
```

### File Structure
- **Data_augmentation**
  - **dataset_splitting.py**: Script to split the dataset into training, validation, and testing datasets.
  - **dataset_augmentation.py**: Script to perform data augmentation on the training dataset.
  - **merge_datasets.py**: Script to merge the augmented datasets with the original dataset.
  - **checking_status.py**: Script to check the status of the DataLoader.
  - **visualizing_data.py**: Script to visualize images and their corresponding labels.

- **Image_enhancement**
  - **image_enhancement.py**: Script for enhancing image clarity.

- **Main_program**
  - **dataset.py**: Script for dataset preparation.
  - **evaluation.py**: Script for evaluating the trained model on the test dataset.
  - **loss_functions.py**: Custom loss functions used during model training.
  - **model_definition.py**: Model architecture definitions.
  - **run.sh**: Shell script to run the training, evaluation, testing, and validation processes.
  - **testing.py**: Script for testing the trained model.
  - **training.py**: Script for training the model.
  - **validation.py**: Script for validating the model during training.

- **Visualize_training_process**
  - **visualize_training_process.py**: Script to visualize the training process.

## Usage
### Running the Entire Program
- To run the entire program including training, evaluation, testing, and validation, use the provided run.sh script:
```bash
./run.sh
```
- 在背景執行腳本並輸出紀錄檔
- To run the script in the background and output logs to output.log:
```bash
nohup bash run.sh > output.log &
```
### Training the Model
- To train the model:
```bash
python training.py --data_path '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Training_dataset_divided_expand_combined' --val_path '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Validation_dataset' --log_file_path '/home/ttsai/Drone_contest_2/Log_file/training.log' --epochs 50 --batch_size 12 --save_path '/home/ttsai/Drone_contest_2/Save_path_all_test'
```

### Evaluating the Model
- To evaluate the trained model on the validation dataset:
```bash
python evaluation.py --model_path '/home/ttsai/Drone_contest_2/Save_path_all_test/best_model.pth' --test_data_path '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Validation_dataset' --log_file_path '/home/ttsai/Drone_contest_2/Log_file/evaluation.log' --batch_size 4
```

### Testing the Model
- To test the trained model:
```bash
python testing.py --test_root '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Testing_dataset' --model_path '/home/ttsai/Drone_contest_2/Save_path_all_test/best_model.pth' --save_path '/home/ttsai/Drone_contest_2/Results_test/Results'
```

### Validating the Model
- To validate the model:
```bash
python validation.py --mask_root '/home/ttsai/Drone_contest_2/Dataset_for_Contest2/Testing_dataset/labels' --pred_root '/home/ttsai/Drone_contest_2/Results_test/Results'
```

### Directory Structure

```plaintext
.
├── Data_augmentation
    ├── README.md
    ├── dataset_splitting.py
    ├── dataset_augmentation.py
    ├── merge_datasets.py
    ├── checking_status.py
    └── visualizing_data.py
├── Image_enhancement
    ├── README.md
    └── image_enhancement.py
├── Images
├── Main_program
    ├── README.md
    ├── dataset.py
    ├── evaluation.py
    ├── loss_functions.py
    ├── model_definition.py
    ├── run.sh
    ├── testing.py
    ├── training.py
    └── validation.py
├── Visualize_training_process
    ├── README.md
    └── visualize_training_process.py
├── README.md
└── env.yml


