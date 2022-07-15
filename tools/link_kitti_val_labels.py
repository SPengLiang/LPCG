import os
import numpy as np
from tqdm import tqdm
import yaml

kitti_cfg = yaml.load(open('./config/kitti.yaml', 'r'), Loader=yaml.Loader)
kitti_root_dir = kitti_cfg['root_dir']
kitti_merge_data_dir = kitti_cfg['KITTI_merge_data_dir']
mode = kitti_cfg['label_mode']


if __name__ == '__main__':
    val_label_dir = os.path.join(kitti_merge_data_dir, 'label_2')
    dst_label_dir = ['{}/{}/label_2'.format(kitti_root_dir, i) for i in mode]
    dst_label_dir_2 = ['{}/{}/filter_label_2'.format(kitti_root_dir, i) for i in mode]

    all_val_labels = sorted(os.listdir(val_label_dir))
    for dst, dst2 in zip(dst_label_dir, dst_label_dir_2):
        for f in tqdm(all_val_labels):
            os.symlink(os.path.join(val_label_dir, f), os.path.join(dst, f))
            os.symlink(os.path.join(val_label_dir, f), os.path.join(dst2, f))

