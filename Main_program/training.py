import argparse
import logging
import os
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model_definition import build_model
from loss_functions import iou_loss_module, bce_loss_module, ssim_loss_module
from dataset import GeneralDataset
# from torch.optim.lr_scheduler import StepLR

def setup_logging(log_file_path):
    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s:%(message)s')

# 載入預訓練模型的權重
def load_model_weights(model, pretrained_model_path):
    pretrained_dict = torch.load(pretrained_model_path)
    model_dict = model.state_dict()
    # 如果模型是DataParallel，修改key的名稱
    if isinstance(model, nn.DataParallel):
        pretrained_dict = {'module.' + k if not k.startswith('module.') else k: v for k, v in pretrained_dict.items()}
    # 只加載與模型匹配的key
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def train_model(data_path, val_path, log_file_path, epochs, batch_size, save_path):
    setup_logging(log_file_path)
    
    logging.info("Loading data...")
    tr_dataset = GeneralDataset(image_root=os.path.join(data_path, "images"),
                                gt_root=os.path.join(data_path, "labels"),
                                img_size=224, mode="train")
    val_dataset = GeneralDataset(image_root=os.path.join(val_path, "images"),
                                 gt_root=os.path.join(val_path, "labels"),
                                 img_size=224, mode="val")
    
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    logging.info(f"Training dataset size: {len(tr_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    
    model = build_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    pretrained_model_path = os.path.join(save_path, 'best_model.pth')
    if os.path.exists(pretrained_model_path):
        load_model_weights(model, pretrained_model_path)
        logging.info(f'Loaded pretrained model from {pretrained_model_path}')
    else:
        logging.info(f'No pretrained model found at {pretrained_model_path}, training from scratch.')
    
    optimizer = optim.Adam(model.parameters(), lr=0.001) # learning rate=0.001
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00008) # learning rate=0.001、L2正則化=0.00008
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5) #學習率每30輪，乘以0.5
    
    best_val_loss = float('inf')
    val_not_down = 0
    pre_val = float('inf')

    for epoch in range(epochs):
        for param_group in optimizer.param_groups:
            logging.info(f"Learning Rate: {param_group['lr']}")
        
        ## -------------Training stage--------------
        model.train()
        loss_all = 0
        epoch_step = 0
        for images, gts in tqdm(tr_loader):
            optimizer.zero_grad()
            images = images.to(device)
            gts = gts.to(device)

            outputs = model(images)
            loss_iou = iou_loss_module(outputs, gts)
            loss_bce = bce_loss_module(outputs, gts)
            ssim_loss = 1 - ssim_loss_module(outputs, gts)
            loss = loss_iou + loss_bce + ssim_loss
            loss.backward()
            optimizer.step()

            epoch_step += 1
            loss_all += loss.item()

        avg_train_loss = loss_all / epoch_step

        ## -------------Validation stage--------------
        model.eval()
        with torch.no_grad():
            val_loss_all = 0
            val_step = 0

            for images, gts in tqdm(val_loader):
                images = images.to(device)
                gts = gts.to(device)
                outputs = model(images)
                loss_iou = iou_loss_module(outputs, gts)
                loss_bce = bce_loss_module(outputs, gts)
                ssim_loss = 1 - ssim_loss_module(outputs, gts)
                val_loss = loss_iou + loss_bce + ssim_loss

                val_loss_all += val_loss.item()
                val_step += 1

            avg_val_loss = val_loss_all / val_step

        # 儲存最好的權重
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(save_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'New best model saved at epoch {epoch}')

        # 每2個epoch儲存一次權重
        if epoch % 2 == 0:
            epoch_model_path = os.path.join(save_path, f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), epoch_model_path)
            logging.info(f'Model saved at epoch {epoch}')

        # 將結果寫入紀錄檔
        with open(log_file_path, 'a') as log_file:
            log_file.write(f'{epoch}, {loss_all:.4f}, {avg_train_loss:.4f}, {val_loss_all:.4f}, {avg_val_loss:.4f}\n')        
        
        # 更新學習率
        # scheduler.step()
        if avg_val_loss > pre_val:
            val_not_down += 1
            if val_not_down >= 2:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8
                    logging.info(f"Learning rate adjusted to {param_group['lr']}")
        else:
            val_not_down = 0
        
        pre_val = avg_val_loss
        logging.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deep learning model.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the training data")
    parser.add_argument('--val_path', type=str, required=True, help="Path to the validation data")
    # 設定輸出紀錄檔的路徑
    parser.add_argument('--log_file_path', type=str, default='training.log', help="Log file path")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    # 設定權重儲存的路徑
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the model")

    args = parser.parse_args()
    
    train_model(args.data_path, args.val_path, args.log_file_path, args.epochs, args.batch_size, args.save_path)