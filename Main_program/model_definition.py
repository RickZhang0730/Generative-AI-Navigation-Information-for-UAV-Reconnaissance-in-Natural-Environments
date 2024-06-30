import segmentation_models_pytorch as smp

def build_model():
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b7",  # 使用 Efficientnet-B7 作為編碼器
        encoder_weights="imagenet",  # 使用 ImageNet 的預訓練權重
        in_channels=3,  # 模型輸入通道
        classes=1,  # 模型輸出通道(1 表示灰階圖)
        activation="sigmoid"  # 使用 sigmoid 激勵函數
    )
    return model

# 初始化模型
model = build_model()