# save as tools/convert_split_to_filenames.py
import os

def convert(src_txt, dst_txt):
    os.makedirs(os.path.dirname(dst_txt), exist_ok=True)
    with open(src_txt, "r", encoding="utf-8") as f, open(dst_txt, "w", encoding="utf-8") as w:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # 取第一个路径的“文件名”
            fn = os.path.basename(parts[0])
            w.write(fn + "\n")

if __name__ == "__main__":
    base = "./data/2025/All/splits"
    convert(os.path.join(base, "train.txt"), os.path.join(base, "all_train.txt"))
    convert(os.path.join(base, "valid.txt"), os.path.join(base, "all_valid.txt"))
    convert(os.path.join(base, "test.txt"),  os.path.join(base, "all_test.txt"))
    print("Done.")
