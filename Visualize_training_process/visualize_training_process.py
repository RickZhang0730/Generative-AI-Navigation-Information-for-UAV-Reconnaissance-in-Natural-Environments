import pandas as pd
import matplotlib.pyplot as plt

# 確認文件路徑
log_path = '/home/ttsai/Drone_contest_2/Save_path_all/training_log.txt'

# 讀取訓練日誌文件並顯示前幾行
try:
    df = pd.read_csv(log_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(log_path, encoding='latin1')

# 檢查數據框的前幾行
print(df.head())
print(df.columns)  # 檢查列名

# 刪除列名中的空格
df.columns = df.columns.str.strip()

print(df.columns)  # 再次檢查列名，確認空格已刪除

# 設置視覺化樣式
plt.style.use('seaborn-darkgrid')

# 創建一個圖形
fig, ax = plt.subplots(figsize=(10, 6))

# 確認列名並繪製訓練和驗證損失
if 'avg_train_loss' in df.columns and 'avg_val_loss' in df.columns:
    ax.plot(df['Epoch'], df['avg_train_loss'], label='Training Loss', marker='o')
    ax.plot(df['Epoch'], df['avg_val_loss'], label='Validation Loss', marker='o')
else:
    print("The expected columns 'avg_train_loss' and 'avg_val_loss' were not found in the log file.")
    print(f"Columns found: {df.columns}")
    exit()

# 設置標籤和標題
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss Over Epochs')

# 顯示圖例
ax.legend()

# 顯示圖形
plt.show()

# 保存圖表為PNG文件
output_path = '/home/ttsai/Drone_contest_2/training_validation_loss/training_validation_loss_3.png'
fig.savefig(output_path, format='png')
print(f"Figure saved as {output_path}")
