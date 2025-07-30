import pandas as pd
import os

csv_path = r'D:\kaggle_pottery\data\h690\jd_sherds_info.csv'
# Only use English column names
use_cols = ["image_id" , "sherd_id" , "unit" , "part" , "type" , "image_side" , "image_id_original"]
df = pd.read_csv(csv_path , usecols = use_cols , encoding="gbk")

print("Database info : ")

print(df.isnull().sum())

# Fill missing values in 'type' column
df["type"] = df["type"].fillna("unknown") 
print("\nAfter filling missing values:\n")
print(df.isnull().sum())

# Function to display unique values and counts for specified columns
def watch_data () :
    all_types = ["unit" , "part" , "type" , "image_side" ]
    for col in all_types :
        print("=" * 50)
        print(f"Unique values in column '{col}':")
        value_counts = df[col].value_counts(dropna=False)
        print(value_counts)
        print(f"\nNumber of unique values in '{col}': {len(value_counts)}")

def label_code(df , cols) :
    """
    Integer encode specified columns
    """
    df_new = df.copy()
    for col in cols :
        df_new[f"{col}_id"] = pd.Categorical(df_new[col]).codes
    return df_new

cols = ["unit" , "part" , "type" , "image_side"]

save_path = r"D:\kaggle_pottery\data\my\clean_data.csv"


if __name__ == '__main__' :
    df_clean = label_code(df , cols)
    os.makedirs(os.path.dirname(save_path) , exist_ok=True)
    df_clean.to_csv(save_path , index = False)
    print("File save !")