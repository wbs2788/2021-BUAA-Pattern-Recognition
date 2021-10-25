'''
Author: wbs2788
Date: 2021-10-12 23:42:34
LastEditTime: 2021-10-26 00:35:27
LastEditors: wbs2788
Description: 
            get data from current path's folder
            put train set to ..\_data\train\ (different folder means different tags)
            valid set: ..\_data\valid\
FilePath: \Pattern Recognition\2021-BUAA-Pattern-Recognition\pick.py
'''
import os
import random
import shutil


def main():
    pwd = os.path.dirname(os.path.abspath(__file__))
    src_base = pwd
    pwd = os.path.join(pwd, "_data")
    dst2_base = os.path.join(pwd, "train")
    dst_base = os.path.join(pwd, "val")
    dst3_base = os.path.join(pwd, "test")
    if not os.path.exists(pwd):
        os.mkdir(pwd)
    if not os.path.exists(dst2_base):
        os.mkdir(dst2_base)
    if not os.path.exists(dst_base):
        os.mkdir(dst_base)
    if not os.path.exists(dst3_base):
        os.mkdir(dst3_base)    

    for sub_dir in sorted(os.listdir(src_base)):
        if sub_dir == "_data":
            continue
        src_dir = os.path.join(src_base, sub_dir)
        dst_dir = os.path.join(dst_base, sub_dir)
        dst2_dir = os.path.join(dst2_base, sub_dir)
        dst3_dir = os.path.join(dst3_base, sub_dir)
        os.makedirs(dst_dir)
        os.makedirs(dst2_dir)
        os.makedirs(dst3_dir)
        image_names = os.listdir(os.path.join(src_base, sub_dir))
        random.shuffle(image_names)        

        for name in image_names[: 3]:
            shutil.copy2(os.path.join(src_dir, name), dst2_dir)
        shutil.copy2(os.path.join(src_dir, image_names[3]), dst_dir)
        for name in image_names[5:]:
            shutil.copy2(os.path.join(src_dir, name), dst3_dir)

if __name__ == '__main__':
    main() 
