import os
from tqdm import tqdm
import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN
import yaml

import utils

kitti_cfg = yaml.load(open('./config/kitti.yaml', 'r'), Loader=yaml.Loader)
kitti_merge_data_dir = kitti_cfg['KITTI_merge_data_dir']
dst_low_cost_label_dir = os.path.join(kitti_cfg['root_dir'], 'low_cost/label_2')


def obtain_cluster_box(l_3d, l_2d, seg_bbox_path, seg_mask_path, P2, wh_range, mask_conf):
    bbox2d = np.loadtxt(seg_bbox_path).reshape(-1, 5)
    bbox_mask = (np.load(seg_mask_path))['masks']

    _, h, w = bbox_mask.shape
    fov_ind = (l_3d[:, 2] > 0) & (l_2d[:, 0] > 0) & (l_2d[:, 1] > 0) & (l_2d[:, 0] < w-1) & (l_2d[:, 1] < h-1)
    l_3d, l_2d = l_3d[fov_ind], l_2d[fov_ind]

    label = []
    for index, b in enumerate(bbox2d):
        if b[-1] < mask_conf:
            continue

        bbox2d_2 = bbox_mask[index]
        bbox2d_2[bbox2d_2 < 0.7] = 0
        bbox2d_2[bbox2d_2 >= 0.7] = 1

        ind = bbox2d_2[l_2d[:, 1], l_2d[:, 0]].astype(np.bool)
        cam_points = l_3d[ind]

        if len(cam_points) < 10:
            continue

        cluster_index = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1).fit_predict(cam_points)

        cam_points = cam_points[cluster_index > -1]
        cluster_index = cluster_index[cluster_index > -1]

        if len(cam_points) < 10:
            continue

        cluster_set = set(cluster_index[cluster_index > -1])
        cluster_sum = np.array([len(cam_points[cluster_index == i]) for i in cluster_set])
        cam_points = cam_points[cluster_index == np.argmax(cluster_sum)]

        rect = cv.minAreaRect(np.array([(cam_points[:, [0, 2]]).astype(np.float32)]))
        (l_t_x, l_t_z), (w, l), rot = rect

        if w > l:
            w, l = l, w
            rot = 90 + rot

        if w > wh_range[0] and w < wh_range[1] and l > wh_range[2] and l < wh_range[3]:
            rect = ((l_t_x, l_t_z), (w, l), rot)
            box = cv.boxPoints(rect)

            h = np.max(cam_points[:, 1]) - np.min(cam_points[:, 1])
            y_center = np.mean(cam_points[:, 1])
            y = y_center + h / 2

            x, z = np.mean(box[:, 0]), np.mean(box[:, 1])
            Ry = (-(np.pi / 2 - (-rot) / 180 * np.pi)) % (np.pi*2)
            if Ry > np.pi:
                Ry -= np.pi*2
            if Ry < -np.pi:
                Ry += np.pi*2

            c_3d = utils.corner_3d([h, w, l, x, y, z, Ry])
            c_2d = utils.convert_to_2d(c_3d, P2)
            bbox = [np.min(c_2d[:, 0]), np.min(c_2d[:, 1]),
                    np.max(c_2d[:, 0]), np.max(c_2d[:, 1])]

            res = np.array([bbox[0], bbox[1], bbox[2], bbox[3],
                              h, w, l, np.mean(box[:, 0]), y, np.mean(box[:, 1]), Ry])
            res = np.round(res, 2)

            label.append(['Car', '0', '0', '0'] + list(res))
    return np.array(label)


if __name__ == '__main__':
    wh_range = [1.2, 1.8, 3.2, 4.2]
    mask_conf = 0.9

    root_dir = kitti_merge_data_dir
    if not os.path.exists(dst_low_cost_label_dir):
        os.makedirs(dst_low_cost_label_dir)

    calib_dir = os.path.join(root_dir, 'calib')
    lidar_dir = os.path.join(root_dir, 'velodyne')
    seg_box_dir = os.path.join(root_dir, 'seg_bbox')
    seg_mask_dir = os.path.join(root_dir, 'seg_mask')
    train_id_path = os.path.join(root_dir, 'split/train.txt')
    train_id = np.loadtxt(train_id_path, dtype=str)

    for i, v in enumerate(tqdm(train_id)):
        lidar_path = os.path.join(lidar_dir, v+'.bin')
        calib_path = os.path.join(calib_dir, v+'.txt')
        seg_bbox_path = os.path.join(seg_box_dir, v+'.txt')
        seg_mask_path = os.path.join(seg_mask_dir, v+'.npz')
        dst_low_cost_label_path = os.path.join(dst_low_cost_label_dir, v+'.txt')

        # obtain and transform lidar points
        l_3d = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 4])[:, :3]
        calibs = utils.parse_calib('3d', calib_path)
        l_3d = (calibs['l2i'] @ (np.concatenate([l_3d, np.ones_like(l_3d[:, :1])], axis=1)).T).T
        l_2d = (utils.convert_to_2d(l_3d, calibs['P2'])).astype(np.int32)

        if not os.path.exists(seg_bbox_path):
            np.savetxt(dst_low_cost_label_path, np.array([]), fmt='%s')
            continue

        low_cost_label = obtain_cluster_box(l_3d, l_2d, seg_bbox_path, seg_mask_path, calibs['P2'], wh_range, mask_conf)
        np.savetxt(dst_low_cost_label_path, low_cost_label, fmt='%s')