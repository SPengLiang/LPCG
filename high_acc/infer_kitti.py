import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import os
from tqdm import tqdm
import yaml


from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


kitti_cfg = yaml.load(open('../../../config/kitti.yaml', 'r'), Loader=yaml.Loader)
kitti_merge_data_dir = kitti_cfg['KITTI_merge_data_dir']
kitti_velodyne = os.path.join(kitti_merge_data_dir, 'velodyne')
dst_high_acc_label_dir = os.path.join(kitti_cfg['root_dir'], 'high_acc/label_2')


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, to_be_pred_list, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext

        self.sample_file_list = to_be_pred_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    root_dir = kitti_merge_data_dir
    if not os.path.exists(dst_high_acc_label_dir):
        os.makedirs(dst_high_acc_label_dir)

    train_id_path = os.path.join(root_dir, 'split/train.txt')
    train_id = np.loadtxt(train_id_path, dtype=str)

    lidar_list = np.array([os.path.join(root_dir, 'velodyne', i+'.bin') for i in train_id])
    cam2cam_list = np.array([os.path.join(root_dir, 'calib_cam2cam', i+'.txt') for i in train_id])
    vel2cam_list = np.array([os.path.join(root_dir, 'calib_vel2cam', i+'.txt') for i in train_id])


    args, cfg = parse_config()
    args.data_path = dst_high_acc_label_dir

    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        to_be_pred_list=lidar_list, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(demo_dataset)):
            dst_high_acc_label_path = os.path.join(dst_high_acc_label_dir, '{:0>6}.txt'.format(idx))

            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            np.savetxt(dst_high_acc_label_path, np.array([]), fmt='%s')

            trans, vel2cam, P2, size = parse_kitti_calib(cam2cam_list[idx], vel2cam_list[idx])

            cls_ind = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}
            pred_boxes = np.around(pred_dicts[0]['pred_boxes'].cpu().numpy().astype(np.float32), 3)
            pred_scores = np.around(pred_dicts[0]['pred_scores'].cpu().numpy().astype(np.float32), 3)
            pred_labels = np.around(pred_dicts[0]['pred_labels'].cpu().numpy().astype(np.float32), 3)

            if pred_boxes.shape[0] < 1:
                continue

            valid_ind = (np.array([i in cls_ind.keys() for i in pred_labels])) & (pred_scores > 0.7)
            if np.sum(valid_ind) < 1:
                continue

            lidar_format_label = np.concatenate([pred_boxes[:, :3], np.ones((pred_boxes.shape[0], 1))], axis=1)
            loc = np.dot(vel2cam, lidar_format_label.T).T

            l, w, h = pred_boxes[:, 3:4], pred_boxes[:, 4:5], pred_boxes[:, 5:6]
            loc[:, 1:2] += h/2
            hwl = np.concatenate([h, w, l], axis=1)
            ry = (-(pred_boxes[:, 6:] + np.pi / 2)) % (np.pi*2)
            ry[ry > np.pi] -= np.pi*2
            ry[ry < -np.pi] += np.pi*2

            loc, hwl, ry, cls = loc[valid_ind], hwl[valid_ind], ry[valid_ind], pred_labels[valid_ind]

            bbox = []
            for i in range(len(loc)):
                h, w, l = hwl[i]
                x, y, z = loc[i]
                c3d = corner_3d([h, w, l, x, y, z, ry[i]])

                c3d_data = np.concatenate([c3d, np.ones((c3d.shape[0], 1))], axis=1)
                c2d = np.dot(P2, c3d_data.T).T
                c2d[:, :2] /= c2d[:, 2:]
                c2d = c2d[:, :2].astype(np.int32)

                bbox.append([np.min(c2d[:, 0]), np.min(c2d[:, 1]), np.max(c2d[:, 0]), np.max(c2d[:, 1])])
            bbox = np.array(bbox)

            bbox_ind = (bbox[:, 2] > 10) & (bbox[:, 0] < 1220) & (bbox[:, 2] - bbox[:, 0] < 1220/2)

            if np.sum(bbox_ind) > 0:
                cls = [cls_ind[i] for i in cls[bbox_ind]]
                pred = np.concatenate([np.zeros((bbox.shape[0], 3), dtype=np.float32), bbox, hwl, loc, ry], axis=1)
                res = ["{} {} {} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(i, *j)
                       for i, j in zip(cls, pred[bbox_ind])]

                np.savetxt(dst_high_acc_label_path, res, fmt='%s')

    logger.info('Done.')


'''
CUDA_VISIBLE_DEVICES=1 python infer_kitti.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../pv_rcnn_8369.pth --data_path /pvc_user/pengliang/LPCG/data/kitti/kitti_merge/training/velodyne
'''


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def corner_3d(bbox3d_input):
    '''

    Args:
        bbox3d_input: a single bbox list with coordinates. format:(6)  [h, w, l, x, y, z, Ry]

    Returns:
        8 corners of the 3d box. format: narUray(8*3) [h, w, l, x, y, z, Ry]
    '''
    '''
        vertex define

       0  ____ 1 
      4 /|___/|            / z(l)
       / /  / / 5         /
    3 /_/_2/ /           /----->x(w)
      |/___|/            |
    7     6              | y(h)
    '''
    bbox3d = bbox3d_input.copy()

    R = roty(bbox3d[-1])

    # # 3d bounding box dimensions
    h = bbox3d[0]
    w = bbox3d[1]
    l = bbox3d[2]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d_shift = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d = np.transpose(corners_3d_shift) + np.array([bbox3d[3], bbox3d[4], bbox3d[5]])

    return corners_3d


def parse_kitti_calib(calib_cam2cam_path, velo2cam_calib_path):
    with open(velo2cam_calib_path, encoding='utf-8') as f:
        text = f.readlines()
        R = np.array(text[1].split(' ')[1:], dtype=np.float32).reshape(3, 3)
        T = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 1)

        trans = np.concatenate([R, T], axis=1)
        vel2cam = trans.copy()

        trans = np.concatenate([trans, np.array([[0, 0, 0, 1]])], axis=0)

    with open(calib_cam2cam_path, encoding='utf-8') as f:
        text = f.readlines()
        P2 = np.array(text[-9].split(' ')[1:], dtype=np.float32).reshape(3, 4)
        size = np.array(text[-11].split(' ')[1:], dtype=np.float32).reshape(-1)
        R_rect = np.zeros((4, 4))
        R_rect_tmp = np.array(text[8].split(' ')[1:], dtype=np.float32).reshape(3, 3)
        R_rect[:3, :3] = R_rect_tmp
        R_rect[3, 3] = 1

        trans = np.dot(np.dot(P2, R_rect), trans)

    vel2cam = np.dot(R_rect_tmp, vel2cam)

    return trans, vel2cam, P2, size


if __name__ == '__main__':
    main()
