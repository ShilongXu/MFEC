import pandas as pd
import os
from terrain_dataset import TerrainEnergyDataset

def validate_data_folder(folder_path):
    """验证单个数据文件夹的完整性"""
    if not os.path.exists(os.path.join(folder_path, 'DataSum.csv')):
        return False

    try:
        df = pd.read_csv(os.path.join(folder_path, 'DataSum.csv'))
        required_cols = ['voltage', 'current', 'PictureName']
        if not all(col in df.columns for col in required_cols):
            return False, "Null values found"

        # 验证数据范围
        if (df['voltage'] <= 0).any() or (df['current'] == 0).any():
            return False

        # 验证图片存在
        sample_img = df.iloc[0]['PictureName'] + '.jpg'
        if not os.path.exists(os.path.join(folder_path, sample_img)):
            return False

        return True, "Valid"

        print("Checking training data...")
        train_data = TerrainEnergyDataset(root_dir=CONFIG['train_root'])
        print(f"Training samples: {len(train_data)}")

        print("\nChecking test data...")
        test_data = TerrainEnergyDataset(root_dir=TEST_CONFIG['test_root'])
        print(f"Test samples: {len(test_data)}")

    except Exception as e:
        return False, str(e)


# 使用示例
if __name__ == "__main__":
    test_dir = input("Enter test data directory: ").strip()
    print(f"\nValidating data in: {test_dir}")
    for folder in os.listdir(test_dir):
        folder_path = os.path.join(test_dir, folder)
        if os.path.isdir(folder_path):
            valid = validate_data_folder(folder_path)
            print(f"{folder}: {'VALID' if valid else 'INVALID'}")