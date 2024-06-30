import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
from model_definition import build_model
from dataset import TestDataset

def test_model(test_root, model_path, save_path, batch_size=12, img_size=224):
    ts_dataset = TestDataset(image_root=os.path.join(test_root, "images"), img_size=img_size)
    ts_loader = DataLoader(dataset=ts_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model()
    model.load_state_dict(torch.load(model_path))
    # model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for images, names in ts_loader:
            images = images.cuda()
            results = model(images)
            results = F.interpolate(results, size=(240, 428), mode='bilinear', align_corners=False)
            results = results.data.cpu().numpy()
            for idx, name in enumerate(names):
                res = results[idx].squeeze()
                success = cv2.imwrite(os.path.join(save_path, name), res * 255)
                if success:
                    print(f"File saved successfully: {name}")
                else:
                    print(f"Failed to save file: {name}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test a deep learning model.")
    parser.add_argument('--test_root', type=str, required=True, help="Path to the testing data")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the results")
    parser.add_argument('--batch_size', type=int, default=12, help="Batch size for testing")
    parser.add_argument('--img_size', type=int, default=224, help="Input image size")

    args = parser.parse_args()
    
    test_model(args.test_root, args.model_path, args.save_path, args.batch_size, args.img_size)
