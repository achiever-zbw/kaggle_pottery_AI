import pandas as pd
import os

csv_path = r'D:\kaggle陶片拼接\data\h690\jd_sherds_info.csv'
# 只采用英文,不用中文了
use_cols = ["image_id" , "sherd_id" , "unit" , "part" , "type" , "image_side" , "image_id_original"]
df = pd.read_csv(csv_path , usecols = use_cols , encoding="gbk")

print("数据基本信息")
# 检查缺失
print(df.isnull().sum())

# type 缺失值填充
df["type"] = df["type"].fillna("unknown") 
print("\n填充后\n")
print(df.isnull().sum())
# 查看数据类型和包含的种类
def watch_data () :
    all_types = ["unit" , "part" , "type" , "image_side" ]
    for col in all_types :
        print("=" * 50)
        print(f"{col} 的唯一值:")
        value_counts = df[col].value_counts(dropna=False)
        print(value_counts)
        print(f"\n{col} 的唯一值数量: {len(value_counts)}")

def label_code(df , cols) :
    """
    对指定列进行整数编码
    """
    df_new = df.copy()
    for col in cols :
        df_new[f"{col}_id"] = pd.Categorical(df_new[col]).codes
    return df_new

cols = ["unit" , "part" , "type" , "image_side"]

save_path = r"D:\kaggle陶片拼接\data\my\clean_data.csv"


if __name__ == '__main__' :
    df_clean = label_code(df , cols)
    os.makedirs(os.path.dirname(save_path) , exist_ok=True)
    df_clean.to_csv(save_path , index = False)
    print("保存成功")