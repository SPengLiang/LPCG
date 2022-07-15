import os
import numpy as np
from tqdm import tqdm
import yaml

kitti_cfg = yaml.load(open('./config/kitti.yaml', 'r'), Loader=yaml.Loader)
kitti_raw_data_dir = kitti_cfg['raw_data_dir']
data_file_dir = os.path.join(kitti_cfg['data_dir'], 'data_file')


train_3D_mapping_file_path = data_file_dir + '/train_mapping.txt'
kitti_3D_rand_file_path = data_file_dir + '/train_rand.txt'
train_3D_file_path = data_file_dir + '/split/train.txt'
val_3D_file_path = data_file_dir + '/split/val.txt'
save_kitti_raw_file_name = data_file_dir + '/split/train_raw.txt'



def build_all_files():
    raw_file_list = []
    raw_file_velo_list = []
    data_dir = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
    for d1 in data_dir:
        data_dir2 = sorted(os.listdir(os.path.join(kitti_raw_data_dir, d1)))
        for d2 in data_dir2:
            if os.path.exists(os.path.join(kitti_raw_data_dir, d1, d2, 'velodyne_points')):
                im_ind = sorted(os.listdir(os.path.join(kitti_raw_data_dir, d1, d2, 'velodyne_points', 'data')))

                im2 = [os.path.join(kitti_raw_data_dir, d1, d2, 'image_02', 'data', i.replace('bin', 'png')) for i in im_ind]
                im3 = [os.path.join(kitti_raw_data_dir, d1, d2, 'image_03', 'data', i.replace('bin', 'png')) for i in im_ind]
                im_velo = [os.path.join(kitti_raw_data_dir, d1, d2, 'velodyne_points', 'data', i) for i in im_ind]

                im_name = np.concatenate([np.array(im2).reshape(-1, 1), np.array(im3).reshape(-1, 1)], axis=1)
                im_name_velo = np.concatenate([np.array(im2).reshape(-1, 1), np.array(im_velo).reshape(-1, 1)],
                                               axis=1)

                raw_file_list.extend(list(im_name))
                raw_file_velo_list.extend(list(im_name_velo))
    raw_file_list = np.array(raw_file_list)
    raw_file_velo_list = np.array(raw_file_velo_list)
    print('build raw files, done')

    return raw_file_list, raw_file_velo_list


def build_train_val_set():
    train_mapping = np.loadtxt(train_3D_mapping_file_path, dtype=str)
    kitti_rand = np.loadtxt(kitti_3D_rand_file_path, delimiter=',')
    train_3D = np.loadtxt(train_3D_file_path).astype(np.uint16)
    val_3D = np.loadtxt(val_3D_file_path).astype(np.uint16)

    train_3D_mapping = train_mapping[(kitti_rand[train_3D]-1).astype(np.uint16)]
    val_3D_mapping = train_mapping[(kitti_rand[val_3D]-1).astype(np.uint16)]
    train_set = set([i[1] for i in train_3D_mapping])
    val_set = set([i[1] for i in val_3D_mapping])

    print('prepare to remove val scenes in raw data')

    return list(train_set), list(val_set), train_3D_mapping, val_3D_mapping


def build_train_files(name_velo_file, val_set):
    good_ind = np.ones(len(name_velo_file)).astype(np.bool)
    scene_name = np.array([i[0].split('/')[-4] for i in name_velo_file])

    for f in tqdm(sorted(val_set)):
        good_ind[scene_name == f] = False

    train_files = name_velo_file[good_ind]
    np.random.shuffle(train_files)

    print('build train files, done')

    return train_files


def add_calib_file(name_velo_file):
    cam_path_1_list = [os.path.join('/', *(f[0].split('/')[:-4]), 'calib_cam_to_cam.txt') for f in name_velo_file]
    cam_path_2_list = [os.path.join('/', *(f[0].split('/')[:-4]), 'calib_velo_to_cam.txt') for f in name_velo_file]
    add_calib_file = np.concatenate([name_velo_file,
                                     np.array(cam_path_1_list).reshape(-1, 1),
                                     np.array(cam_path_2_list).reshape(-1, 1)], axis=1)

    print('add calib files, done')

    return add_calib_file


if __name__ == '__main__':
    raw_file_list, raw_file_velo_list = build_all_files()
    train_set, val_set, train_3D_mapping, val_3D_mapping = build_train_val_set()
    train_files = build_train_files(raw_file_velo_list, val_set)
    train_add_calib = add_calib_file(train_files)

    np.savetxt(save_kitti_raw_file_name, train_add_calib, fmt='%s')







