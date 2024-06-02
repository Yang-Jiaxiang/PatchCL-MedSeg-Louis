import pandas as pd
from datetime import datetime
import os


def save_loss(
    t_total_loss=0, 
    t_supervised_loss=0, 
    t_contrastive_loss=0, 
    t_consistency_loss=0, 
    t_miou=0,    
    t_accuracy=0,
    t_dice=0,
    v_total_loss=0, 
    v_supervised_loss=0, 
    v_contrastive_loss=0, 
    v_consistency_loss=0, 
    v_miou=0,    
    v_accuracy=0,
    v_dice=0,
    PatchCL_weight=0,
    filename='/tf/PatchCL-MedSeg-pioneeryj/loss_record.csv'
):
    # 创建一个包含损失值和当前时间的字典
    data = {
        't_loss': [t_total_loss], 
        't_supervised_loss':[t_supervised_loss], 
        't_contrastive_loss':[t_contrastive_loss], 
        't_consistency_loss':[t_consistency_loss], 
        't_miou':[t_miou],
        't_accuracy':[t_accuracy],
        't_dice':[t_dice],
        'v_loss': [v_total_loss], 
        'v_supervised_loss':[v_supervised_loss], 
        'v_contrastive_loss':[v_contrastive_loss], 
        'v_consistency_loss':[v_consistency_loss], 
        'v_miou':[v_miou],
        'v_accuracy':[v_accuracy],
        'v_dice':[v_dice],
        'PatchCL_weight':[PatchCL_weight],
        'time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }
    
    # 检查文件是否存在
    if not os.path.exists(filename):
        # 如果文件不存在，创建一个新的 DataFrame，并保存为 CSV
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Created new file and saved data: {filename}")
    else:
        # 如果文件已存在，加载文件，添加新数据，并保存
        df = pd.read_csv(filename)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(filename, index=False)

def check_loss_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"File {filename} has been deleted.")
    else:
        print(f"File {filename} does not exist in the directory.")

