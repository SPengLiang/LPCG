import os
import sys
import numpy as np
from tqdm import tqdm
import yaml


kitti_cfg = yaml.load(open('./config/kitti.yaml', 'r'), Loader=yaml.Loader)
kitti_dir = kitti_cfg['data_dir']
dst_dir = kitti_cfg['KITTI_merge_data_dir']
root_3d_dir = kitti_cfg['KITTI3D_data_dir']

source_file = kitti_dir + '/data_file/split/train_raw.txt'
val_id = np.loadtxt(kitti_dir + '/data_file/split/val.txt', dtype=int)


if __name__ == '__main__':
    raw_info = np.loadtxt(source_file, dtype=str)
    img_list = raw_info[:, 0]
    lidar_list = raw_info[:, 1]
    cam2cam = raw_info[:, 2]
    vel2cam = raw_info[:, 3]


    split_dst_dir = os.path.join(dst_dir, 'split')
    label_dst_dir = os.path.join(dst_dir, 'label_2')
    image_dst_dir = os.path.join(dst_dir, 'image_2')
    lidar_dst_dir = os.path.join(dst_dir, 'velodyne')
    calib_dst_dir = os.path.join(dst_dir, 'calib')
    cam2cam_dst_dir = os.path.join(dst_dir, 'calib_cam2cam')
    vel2cam_dst_dir = os.path.join(dst_dir, 'calib_vel2cam')
    all_dst_dir = [split_dst_dir, label_dst_dir, image_dst_dir, lidar_dst_dir, calib_dst_dir, cam2cam_dst_dir, vel2cam_dst_dir]
    for d in all_dst_dir:
        if not os.path.exists(d):
            os.makedirs(d)


    for i, v in enumerate(tqdm(raw_info)):
        base_name = '{:0>6}'.format(i)
        os.symlink(img_list[i], os.path.join(image_dst_dir, base_name+'.png'))
        os.symlink(lidar_list[i], os.path.join(lidar_dst_dir, base_name+'.bin'))
        os.symlink(cam2cam[i], os.path.join(cam2cam_dst_dir, base_name+'.txt'))
        os.symlink(vel2cam[i], os.path.join(vel2cam_dst_dir, base_name+'.txt'))


    all_f = sorted(os.listdir(cam2cam_dst_dir))
    for f in tqdm(all_f):
        calib_cam2cam_path = os.path.join(cam2cam_dst_dir, f)
        with open(calib_cam2cam_path, encoding='utf-8') as fw:
            text = fw.readlines()
            P2 = text[-9].split(' ')[1:]
            R_rect = text[8].split(' ')[1:]

        velo2cam_calib_path = os.path.join(vel2cam_dst_dir, f)
        with open(velo2cam_calib_path, encoding='utf-8') as fw:
            text = fw.readlines()
            R = np.array(text[1].split(' ')[1:], dtype=np.float32).reshape(3, 3)
            T = np.array(text[2].split(' ')[1:], dtype=np.float32).reshape(3, 1)

            Tr_velo_to_cam = np.concatenate([R, T], axis=1).reshape(-1)
            Tr_velo_to_cam = [str(i) for i in Tr_velo_to_cam]
            Tr_velo_to_cam[-1] += '\n'

        write_calib_3d_path = os.path.join(calib_dst_dir, f)
        # note that currently we ignore P0, P1, P3 and Tr_imu_to_velo!
        with open(write_calib_3d_path, 'w') as fw:
            fw.writelines("P0: " + ' '.join(P2))
            fw.writelines("P1: " + ' '.join(P2))
            fw.writelines("P2: " + ' '.join(P2))
            fw.writelines("P3: " + ' '.join(P2))
            fw.writelines("R0_rect: " + ' '.join(R_rect))
            fw.writelines("Tr_velo_to_cam: " + ' '.join(Tr_velo_to_cam))
            fw.writelines("Tr_imu_to_velo: " + ' '.join(Tr_velo_to_cam))


    cur_id_len = len(os.listdir(calib_dst_dir))
    train_len = cur_id_len

    # for kitti val
    val_len = len(val_id)
    for i, v in enumerate(tqdm(val_id)):
        dst_name = '{:0>6}'.format(i + cur_id_len)
        source_name = '{:0>6}'.format(v)

        os.symlink(os.path.join(root_3d_dir, 'label_2', source_name+'.txt'),
                   os.path.join(label_dst_dir, dst_name+'.txt'))
        os.symlink(os.path.join(root_3d_dir, 'image_2', source_name+'.png'),
                   os.path.join(image_dst_dir, dst_name+'.png'))
        os.symlink(os.path.join(root_3d_dir, 'velodyne', source_name+'.bin'),
                   os.path.join(lidar_dst_dir, dst_name+'.bin'))
        os.symlink(os.path.join(root_3d_dir, 'calib', source_name+'.txt'),
                   os.path.join(calib_dst_dir, dst_name+'.txt'))


    train_id = np.array(['{:0>6}'.format(i) for i in np.arange(train_len)])
    val_id = np.array(['{:0>6}'.format(i) for i in np.arange(train_len, train_len+val_len)])
    np.savetxt(os.path.join(split_dst_dir, 'train.txt'), train_id, fmt='%s')
    np.savetxt(os.path.join(split_dst_dir, 'val.txt'), val_id, fmt='%s')

