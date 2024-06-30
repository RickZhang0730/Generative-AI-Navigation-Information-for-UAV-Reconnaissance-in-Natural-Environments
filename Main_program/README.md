# Main Program for UAV Image Processing and Segmentation

This folder is designed for processing and segmenting UAV images. It includes functionalities for data augmentation, model training, validation, and evaluation. Through these functionalities, users can preprocess UAV-captured images, train deep learning models, evaluate model performance on the validation set, and ultimately assess the model's effectiveness on the test set.

### File Structure

- **dataset.py**: Script for dataset preparation.
- **evaluation.py**: Script for evaluating the trained model on the test dataset.
- **loss_functions.py**: Custom loss functions used during model training.
- **model_definition.py**: Model architecture definitions.
- **run.sh**: Shell script to run the training, evaluation, testing, and validation processes.
- **testing.py**: Script for testing the trained model.
- **training.py**: Script for training the model.
- **validation.py**: Script for validating the model during training.

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
├── dataset.py
├── evaluation.py
├── loss_functions.py
├── model_definition.py
├── run.sh
├── testing.py
├── training.py
└── validation.py

