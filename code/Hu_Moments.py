"""基于Hu矩的陶片轮廓匹配脚本

功能说明：
- 本脚本通过计算陶片轮廓的Hu矩特征，衡量轮廓形状相似度，实现陶片之间的相似轮廓匹配。
- 使用清理后的CSV（clean_data.csv）中记录的单位(unit)信息，将陶片按单位分组，确保只在同单位内进行匹配。
- 计算每个轮廓与同单位其他轮廓的相似度，挑选距离最近的top-k（默认10）作为匹配结果。
- 匹配结果保存为CSV文件，方便后续分析与使用。
"""
import cv2
import numpy as np
import os
import csv
import pandas as pd
from collections import defaultdict

class HuMomentsMatcher:
    def __init__(self, contours_dir, filename_unit_map):
        self.contours_dir = contours_dir
        self.contour_files = [f for f in os.listdir(contours_dir) if f.endswith('.npy')]
        self.hu_moments = {}
        self.filename_unit_map = filename_unit_map

    def compute_hu_moments(self, contour):
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten()
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        return hu

    def load_and_compute_all(self):
        print(f"Loading and computing Hu moments for {len(self.contour_files)} contours in {self.contours_dir} ...")
        for idx, file in enumerate(self.contour_files):
            path = os.path.join(self.contours_dir, file)
            contour = np.load(path)
            hu = self.compute_hu_moments(contour)
            self.hu_moments[file] = hu
            if (idx + 1) % 50 == 0 or (idx + 1) == len(self.contour_files):
                print(f"  Processed {idx + 1}/{len(self.contour_files)} contours")

    def similarity(self, hu1, hu2):
        return np.linalg.norm(hu1 - hu2)

    def get_topk_matches(self, k=10):
        print(f"Computing top-{k} matches for each file ...")
        topk_dict = defaultdict(list)
        files = list(self.hu_moments.keys())

        for i in range(len(files)):
            f1 = files[i]
            hu1 = self.hu_moments[f1]
            unit1 = self.filename_unit_map.get(f1, None)
            if unit1 is None:
                continue

            distances = []
            for j in range(len(files)):
                if i == j:
                    continue
                f2 = files[j]
                unit2 = self.filename_unit_map.get(f2, None)
                if unit2 is None or unit1 != unit2:
                    continue
                dist = self.similarity(hu1, self.hu_moments[f2])
                distances.append((f2, dist))

            distances.sort(key=lambda x: x[1])
            topk_dict[f1] = [f for f, _ in distances[:k]]

            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"  Completed top-{k} for {i + 1}/{len(files)} files")

        return topk_dict


def save_topk_matches(topk_dict, k, output_csv):
    print(f"Saving top-{k} matches per file to {output_csv} ...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file'] + [f'match{i+1}' for i in range(k)])
        for file, matches in topk_dict.items():
            padded_matches = matches + [''] * (k - len(matches))
            writer.writerow([file] + padded_matches)
    print("Top-k match file saved.")


def build_filename_unit_map(clean_csv_path, contours_dir):
    df = pd.read_csv(clean_csv_path)
    filename_unit_map = {}

    for filename in os.listdir(contours_dir):
        if not filename.endswith('.npy'):
            continue
        name_no_ext = filename[:-4]
        unit_row = df[df['image_id'] == name_no_ext]
        if not unit_row.empty:
            filename_unit_map[filename] = unit_row.iloc[0]['unit']
        else:
            filename_unit_map[filename] = None
            print(f"Warning: image_id {name_no_ext} not found in clean_data.csv")

    return filename_unit_map


if __name__ == '__main__':
    ex_dir = r"D:\kaggle_pottery\data\h690\contours_ex"
    in_dir = r"D:\kaggle_pottery\data\h690\contours_in"
    csv_dir = r"D:\kaggle_pottery\data\csv"
    clean_csv_path = r"D:\kaggle_pottery\data\my\clean_data.csv"

    # 外轮廓 top10 匹配
    ex_map = build_filename_unit_map(clean_csv_path, ex_dir)
    matcher_ex = HuMomentsMatcher(ex_dir, ex_map)
    matcher_ex.load_and_compute_all()
    topk_ex = matcher_ex.get_topk_matches(k=10)
    save_topk_matches(topk_ex, 10, os.path.join(csv_dir, 'exterior_top10_matches.csv'))

    # 内轮廓 top10 匹配
    in_map = build_filename_unit_map(clean_csv_path, in_dir)
    matcher_in = HuMomentsMatcher(in_dir, in_map)
    matcher_in.load_and_compute_all()
    topk_in = matcher_in.get_topk_matches(k=10)
    save_topk_matches(topk_in, 10, os.path.join(csv_dir, 'interior_top10_matches.csv'))

    print("Top-10 matching and saving completed.")
