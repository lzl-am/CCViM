import argparse
import cProfile as profile
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

# 解决找不到metrics.stats_utils的问题
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from metrics.stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_dice_2,
    get_fast_pq,
    remap_label,
    pair_coordinates
)


def consep_type(anno):
    ann_type = anno["type_map"]
    inst_type = anno['inst_type']

    # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
    # If own dataset is used, then the below may need to be modified
    ann_type[(ann_type == 3) | (ann_type == 4)] = 3
    ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

    inst_type[(inst_type == 3) | (inst_type == 4)] = 3
    inst_type[(inst_type == 5) | (inst_type == 6) | (inst_type == 7)] = 4

    anno["type_map"] = ann_type
    anno["inst_type"] = inst_type

    return anno


def run_nuclei_type_stat(pred_dir, true_dir, type_uid_list=None, exhaustive=True):
    """GT must be exhaustively annotated for instance location (detection).
    细胞核类型分类评估

    Args:
        true_dir, pred_dir: Directory contains .mat annotation for each image.
                            Each .mat must contain:
                    --`inst_centroid`: Nx2, contains N instance centroid
                                       of mass coordinates (X, Y)
                    --`inst_type`    : Nx1: type of each instance at each index
                    `inst_centroid` and `inst_type` must be aligned and each
                    index must be associated to the same instance
        type_uid_list : list of id for nuclei type which the score should be calculated.
                        Default to `None` means available nuclei type in GT.
        exhaustive : Flag to indicate whether GT is exhaustively labelled
                     for instance types

    """
    # 获取所有预测文件，按名称排序（确保GT与预测文件一一对应）
    file_list = glob.glob(pred_dir + "*.mat")
    file_list.sort()  # ensure same order [1]

    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point

    # 遍历每个文件，处理GT与预测数据
    for file_idx, filename in enumerate(file_list[:]):
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]

        true_info = sio.loadmat(os.path.join(true_dir, basename + ".mat"))
        true_info = consep_type(true_info)
        # dont squeeze, may be 1 instance exist
        true_centroid = (true_info["inst_centroid"]).astype("float32")
        true_inst_type = (true_info["inst_type"]).astype("int32")

        if true_centroid.shape[0] != 0:
            true_inst_type = true_inst_type[:, 0]
        else:  # no instance at all
            true_centroid = np.array([[0, 0]])
            true_inst_type = np.array([0])

        # * for converting the GT type in CoNSeP
        # true_inst_type[(true_inst_type == 3) | (true_inst_type == 4)] = 3
        # true_inst_type[(true_inst_type == 5) | (true_inst_type == 6) | (true_inst_type == 7)] = 4

        pred_info = sio.loadmat(os.path.join(pred_dir, basename + ".mat"))
        # dont squeeze, may be 1 instance exist
        pred_centroid = (pred_info["inst_centroid"]).astype("float32")
        pred_inst_type = (pred_info["inst_type"]).astype("int32")

        if pred_centroid.shape[0] != 0:
            pred_inst_type = pred_inst_type[:, 0]
        else:  # no instance at all
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroid, pred_centroid, 12
        )  # 根据质心坐标的距离来配对True的实例和Pred的实例的Index

        # * Aggreate information
        # get the offset as each index represent 1 independent instance
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        true_inst_type_all.append(true_inst_type)
        pred_inst_type_all.append(pred_inst_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)

    paired_all = np.concatenate(paired_all, axis=0)  # 0 是ture的细胞核index，1 是pred的细胞核index
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    ###
    def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        """定义类型F1分数计算函数"""
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        if not exhaustive:
            ignore = (paired_true == -1).sum()
            fp_dt -= ignore

        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()

        f1_type = (2 * (tp_dt + tn_dt)) / (
                2 * (tp_dt + tn_dt)
                + w[0] * fp_dt
                + w[1] * fn_dt
                + w[2] * fp_d
                + w[3] * fn_d
        )
        return f1_type

    # overall
    # * quite meaningless for not exhaustive annotated dataset
    w = [1, 1]
    # 配对成功的实例数（检测正确）
    tp_d = paired_pred_type.shape[0]
    # 未配对的预测实例（检测假阳性）
    fp_d = unpaired_pred_type.shape[0]
    # 未配对的GT实例（检测假阴性）
    fn_d = unpaired_true_type.shape[0]

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    if not exhaustive:
        ignore = (paired_true_type == -1).sum()
        fp_fn_dt -= ignore

    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
    # 检测F1（衡量实例匹配准确性）
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    # 计算每个类型的F1分数
    # 类型F1的权重（类型错误权重高于检测错误）
    w = [2, 2, 1, 1]

    if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()

    # 结果列表：[检测F1, 类型准确率, 各类型F1]
    results_list = [f1_d, acc_type]
    for type_uid in type_uid_list:
        f1_type = _f1_type(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w,
        )
        results_list.append(f1_type)

    # 打印结果（按CoNSeP类型顺序：背景、其他、炎症、上皮、梭形）
    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    print(f'f1_d: {results_list[0]}\n'
          f'acc_type: {results_list[1]}\n'
          f'other: {results_list[2]}\n'
          f'inflammary: {results_list[3]}\n'
          f'epithelical: {results_list[4]}\n'
          f'spindle-shaped: {results_list[5]}\n')
    # print('fd, acc, other, inflammary, epithelical, spindle-shaped:',np.array(results_list))
    return


def run_nuclei_inst_stat(pred_dir, true_dir, print_img_stats=False, ext=".mat"):
    """细胞核实例分割评估"""
    # print stats of each image
    print(pred_dir)

    # 步骤1：获取预测文件列表，按名称排序（确保GT与预测对应）
    file_list = glob.glob("%s/*%s" % (pred_dir, ext))
    file_list.sort()  # ensure same order

    # 步骤2：初始化指标列表（存储每个文件的指标）
    # 指标顺序：[Dice, AJI, DQ, SQ, PQ, AJI+]
    metrics = [[], [], [], [], [], []]

    # 步骤3：遍历每个文件，计算指标
    for filename in file_list[:]:
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]

        # 加载GT和预测的实例掩码
        true = sio.loadmat(os.path.join(true_dir, basename + ".mat"))
        true = (true["inst_map"]).astype("int32")
        pred = sio.loadmat(os.path.join(pred_dir, basename + ".mat"))
        pred = (pred["inst_map"]).astype("int32")

        # to ensure that the instance numbering is contiguous
        # 步骤4：重置实例索引（确保索引连续，避免因编号断层影响评估）
        # 例：若GT实例索引为[1,3,5]，remap后变为[1,2,3]
        pred = remap_label(pred, by_size=False)
        true = remap_label(true, by_size=False)

        # 步骤5：调用自定义函数计算分割指标
        pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]  # PQ相关指标（DQ, SQ, PQ）
        metrics[0].append(get_dice_1(true, pred))            # Dice系数（区域重合度）
        metrics[1].append(get_fast_aji(true, pred))          # AJI（平均交并比，考虑实例匹配）
        metrics[2].append(pq_info[0])                        # DQ（检测质量：实例匹配准确性）
        metrics[3].append(pq_info[1])                        # SQ（分割质量：区域重合准确性）
        metrics[4].append(pq_info[2])                        # PQ（综合质量：DQ×SQ）
        metrics[5].append(get_fast_aji_plus(true, pred))     # AJI+（改进版AJI，更鲁棒）

        # 可选：打印单张图像的指标（用于定位异常样本）
        if print_img_stats:
            print(basename, end="\t")
            for scores in metrics:
                print("%f " % scores[-1], end="  ")
            print()
    ####

    # 步骤6：计算所有文件的平均指标并打印
    metrics = np.array(metrics)
    metrics_avg = np.mean(metrics, axis=-1)
    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    print(f'dice_1 {metrics_avg[0]}\n'
          f'fast_aji {metrics_avg[1]}\n'
          f'dq {metrics_avg[2]}\n'
          f'sq {metrics_avg[3]}\n'
          f'pq {metrics_avg[4]}\n'
          f'fast_aji_plus {metrics_avg[5]}\n'
          )
    print(metrics_avg)
    metrics_avg = list(metrics_avg)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="mode to run the measurement,"
             "`type` for nuclei instance type classification or"
             "`instance` for nuclei instance segmentation",
        nargs="?",
        default="instance",
        const="instance",
    )
    parser.add_argument(
        "--pred_dir", help="point to output dir", nargs="?",
        default="/seu_share/home/220232363/baseline/CCViM/CCViM-nuclei/CPM17/infer_output/CCM_UNET_2025-11-03_18-04-34/mat", const=""
    )
    parser.add_argument(
        "--true_dir", help="point to ground truth dir", nargs="?",
        default="/seu_share/home/220232363/data/cpm17/test/Labels/", const=""
    )
    args = parser.parse_args()

    if args.mode == "instance":
        run_nuclei_inst_stat(args.pred_dir, args.true_dir, print_img_stats=False)
    if args.mode == "type":
        run_nuclei_type_stat(args.pred_dir, args.true_dir)
