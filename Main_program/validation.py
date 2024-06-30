import os
import cv2
from py_sod_metrics import FmeasureV2, FmeasureHandler

def validate(mask_root, pred_root):
    # 計算 Mean F-measure
    mask_name_list = sorted(os.listdir(mask_root))
    FMv2 = FmeasureV2(
        metric_handlers={
            "fm": FmeasureHandler(with_dynamic=True, with_adaptive=False, beta=0.3),
        }
    )

    for i, mask_name in enumerate(mask_name_list):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FMv2.step(pred=pred, gt=mask)

    fmv2_results = FMv2.get_results()

    results = {
        "meanfm": fmv2_results["fm"]["dynamic"].mean()
    }

    print(results)
    print("Eval finished!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate model predictions.")
    # 設定模型預測結果的路徑
    parser.add_argument('--mask_root', type=str, required=True, help="Path to the ground truth masks") 
    # 設定 ground truth 路徑
    parser.add_argument('--pred_root', type=str, required=True, help="Path to the predicted masks")

    args = parser.parse_args()
    
    validate(args.mask_root, args.pred_root)
