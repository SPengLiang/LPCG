import os
import numpy as np
from tqdm import tqdm
import yaml

kitti_cfg = yaml.load(open('./config/kitti.yaml', 'r'), Loader=yaml.Loader)
kitti_root_dir = kitti_cfg['root_dir']
kitti_merge_data_dir = kitti_cfg['KITTI_merge_data_dir']
mode = kitti_cfg['label_mode']
target_cls = ['Car']

if __name__ == '__main__':
    label_dir = ['{}/{}/label_2'.format(kitti_root_dir, i) for i in mode]
    dst_filter_label_dir = ['{}/{}/filter_label_2'.format(kitti_root_dir, i) for i in mode]
    for d in dst_filter_label_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    root_dir = kitti_merge_data_dir
    train_id_path = os.path.join(root_dir, 'split/train.txt')
    train_id = np.loadtxt(train_id_path, dtype=str)

    filter_train_id_path = [os.path.join(root_dir, 'split/train_{}.txt'.format(i)) for i in mode]

    for l_d, dst_l_d, filter_id in zip(label_dir, dst_filter_label_dir, filter_train_id_path):
        id_list = []
        for id in tqdm(train_id):
            cur_label = np.loadtxt(os.path.join(l_d, id+'.txt'), dtype=str).reshape(-1, 15)
            if cur_label.shape[0] < 1:
                continue

            cur_label[cur_label[:, 4].astype(np.float32) < 0] = 0
            cur_label[cur_label[:, 6].astype(np.float32) < 0] = 0

            cur_label_ind = [i[0] in target_cls for i in cur_label]
            if np.sum(cur_label_ind) < 1:
                continue

            id_list.append(id)
            np.savetxt(os.path.join(dst_l_d, id+'.txt'), cur_label[cur_label_ind], fmt='%s')
        np.savetxt(filter_id, id_list,  fmt='%s')
