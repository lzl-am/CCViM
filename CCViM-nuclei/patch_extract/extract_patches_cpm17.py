"""extract_patches_consep.py

Patch extraction script.
"""

import re
import glob
import os
import tqdm
import pathlib

import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import mkdir

from dataset import get_dataset

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = False

    # patch 大小为 540x540 像素
    win_size = [540, 540]
    # 步长为 164x164 ，patch 间会有重叠，避免边界信息丢失
    step_size = [164, 164]
    # mirror 表示边界处用镜像填充，避免补丁包含无效区域
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset_CKC.py
    dataset_name = "cpm17"
    save_root = "/opt/data/private/zhuyun/dataset/cpm17/processed/"

    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "train": {
            "img": (".png", "/opt/data/private/zhuyun/dataset/cpm17/train/Images/"),
            "ann": (".mat", "/opt/data/private/zhuyun/dataset/cpm17/train/Labels/"),
        },
        "valid": {
            "img": (".png", "/opt/data/private/zhuyun/dataset/cpm17/test/Images/"),
            "ann": (".mat", "/opt/data/private/zhuyun/dataset/cpm17/test/Labels/"),
        },
    }

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    # 获取CPM17数据集的解析器（根据数据集类型，提供图像和标注的加载方法）
    parser = get_dataset(dataset_name)
    # 初始化补丁提取器，传入补丁大小和步长
    xtractor = PatchExtractor(win_size, step_size)

    # 批量提取补丁
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        # 定义当前数据集 patch 的保存路径
        out_dir = "%s/%s/%s/%dx%d_%dx%d/" % (
            save_root,
            dataset_name,
            split_name,
            win_size[0], # 540
            win_size[1],
            step_size[0],  # 164
            step_size[1],
        )
        # 查找所有标注文件并排序
        file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
        file_list.sort()  # ensure same ordering across platform

        mkdir(out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        # 遍历每个标注文件（对应一张图像）
        for file_idx, file_path in enumerate(file_list):
            # 获取文件名（不含扩展名），用于匹配对应的图像文件
            base_name = pathlib.Path(file_path).stem

            # 加载对应的图像和标注
            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            )

            # 将图像和标注在最后一个维度拼接（合并为一个数组，方便同步提取补丁）
            # 例如：img.shape=(H, W, 3)（RGB），ann.shape=(H, W, K)（K为标注通道数）→ 拼接后为(H, W, 3+K)
            img = np.concatenate([img, ann], axis=-1)

            # 提取patch：使用 mirror 填充边界（extract_type指定），返回所有子补丁的列表
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            # 保存每个 patch 为 .npy 文件
            for idx, patch in enumerate(sub_patches):
                # 文件名格式：原图名称_patch序号.npy（如"case1_000.npy"）
                np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()
