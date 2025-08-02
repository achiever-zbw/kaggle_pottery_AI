"""将 labelme 标注的 json 文件转化为 mask_png"""
import os
import subprocess
import shutil

input_dir = "D:/kaggle_pottery/data/unet_labels"

# 遍历所有 json 文件
for file in os.listdir(input_dir):
    if file.endswith(".json"):
        json_path = os.path.join(input_dir, file)
        base_name = os.path.splitext(file)[0]
        temp_output = os.path.join(input_dir, base_name + "_json") 

        print(f"正在处理：{file}")

        try:
            # 调用 labelme_json_to_dataset 命令
            subprocess.run(["labelme_json_to_dataset", json_path], check=True)

            # 找到 label.png 并重命名保存为 mask
            label_src = os.path.join(temp_output, "label.png")
            mask_dst = os.path.join(input_dir, f"{base_name}_mask.png")

            if os.path.exists(label_src):
                shutil.move(label_src, mask_dst)
                print(f"生成 mask：{mask_dst}")
            else:
                print(f"未找到 label.png，跳过：{file}")

        except subprocess.CalledProcessError as e:
            print(f"转换失败：{file}\n错误信息：{e}")

        finally:
            # 清理临时文件夹和 json 文件
            if os.path.exists(temp_output):
                shutil.rmtree(temp_output)
            os.remove(json_path)

print("\n所有 JSON 已转换为 mask，JSON 文件已删除。")
