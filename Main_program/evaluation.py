import argparse
import logging
import torch
import os
from torch.utils.data import DataLoader
from model_definition import build_model
from loss_functions import iou_loss_module, bce_loss_module, ssim_loss_module
from dataset import GeneralDataset

def setup_logging(log_file_path):
    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s:%(message)s')

def evaluate_model(model_path, test_data_path, log_file_path, batch_size=32):
    setup_logging(log_file_path)
    
    logging.info("Loading model...")
    model = build_model()
    model.load_state_dict(torch.load(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    logging.info("Loading test data...")
    test_dataset = GeneralDataset(image_root=os.path.join(test_data_path, "images"),
                                  gt_root=os.path.join(test_data_path, "labels"),
                                  img_size=224, mode="val")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    logging.info(f"Test dataset size: {len(test_dataset)}")
    
    model.eval()
    total_loss = 0.0
    total_iou_loss = 0.0
    total_bce_loss = 0.0
    total_ssim_loss = 0.0
    with torch.no_grad():
        for images, gts in test_loader:
            images = images.to(device)
            gts = gts.to(device)
            
            outputs = model(images)
            loss_iou = iou_loss_module(outputs, gts)
            loss_bce = bce_loss_module(outputs, gts)
            ssim_loss = 1 - ssim_loss_module(outputs, gts)
            loss = loss_iou + loss_bce + ssim_loss
            
            total_loss += loss.item()
            total_iou_loss += loss_iou.item()
            total_bce_loss += loss_bce.item()
            total_ssim_loss += ssim_loss.item()
    
    avg_loss = total_loss / len(test_loader)
    avg_iou_loss = total_iou_loss / len(test_loader)
    avg_bce_loss = total_bce_loss / len(test_loader)
    avg_ssim_loss = total_ssim_loss / len(test_loader)
    
    logging.info(f"Test Loss: {avg_loss}, IOU Loss: {avg_iou_loss}, BCE Loss: {avg_bce_loss}, SSIM Loss: {avg_ssim_loss}")
    print(f"Test Loss: {avg_loss}, IOU Loss: {avg_iou_loss}, BCE Loss: {avg_bce_loss}, SSIM Loss: {avg_ssim_loss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a deep learning model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--test_data_path', type=str, required=True, help="Path to the test data")
    parser.add_argument('--log_file_path', type=str, default='evaluation.log', help="Log file path")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_data_path, args.log_file_path, args.batch_size)
