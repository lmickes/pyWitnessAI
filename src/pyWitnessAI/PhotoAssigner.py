import os
import shutil
import pandas as pd
import re

def organize_photos(photos_directory, data_file, label_column):
    # 自动读取 CSV 或 Excel
    file_extension = os.path.splitext(data_file)[1].lower()
    if file_extension == '.csv':
        df = pd.read_csv(data_file)
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(data_file)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx/.xls")

    # 自动找到照片编号列
    number_col = None
    for col in df.columns:
        # 通过正则匹配列中是否存在“xxx_数字”或“数字.jpg”这种格式
        if df[col].astype(str).str.contains(r'\d+').any():
            number_col = col
            break

    if number_col is None:
        raise ValueError("No column with photo numbers detected.")

    # 提取照片编号 -> e.g., 从 similarity_1.jpg -> 1
    df['photo_number'] = df[number_col].astype(str).str.extract(r'(\d+)')

    # 读取照片文件夹下所有照片文件，并提取编号
    photo_files = [f for f in os.listdir(photos_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    photo_dict = {}
    for photo in photo_files:
        match = re.search(r'(\d+)', photo)
        if match:
            photo_dict[match.group(1)] = photo  # 用编号作为key

    # 提取标签列
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in the data file.")

    # 获取文档名作为文件夹前缀
    doc_name = os.path.splitext(os.path.basename(data_file))[0]

    # 开始归类
    for _, row in df.iterrows():
        num = str(row['photo_number'])
        label = row[label_column]

        if num in photo_dict:
            photo_name = photo_dict[num]
            dest_dir = os.path.join(photos_directory, f"{label_column}_{label}")
            os.makedirs(dest_dir, exist_ok=True)
            source_path = os.path.join(photos_directory, photo_name)
            dest_path = os.path.join(dest_dir, photo_name)
            shutil.copy(source_path, dest_path)
        else:
            print(f"Warning: Photo with number {num} not found in {photos_directory}.")

    print("Photo organization completed.")

# Example:
# organize_photos("D:/PhD/Studies/2ndStudy/easyVideoMorgan/Fillerpool",
#                 "D:/PhD/Studies/2ndStudy/similarity_data_labeled.csv",
#                 "similarity_group_to_perp_facenet512")