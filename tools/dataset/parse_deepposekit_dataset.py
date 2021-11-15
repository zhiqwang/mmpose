import cv2
import h5py
import json
import numpy as np
import os
import time

np.random.seed(0)


def save_coco_anno(keypoints_all,
                   annotated_all,
                   imgs_all,
                   keypoints_info,
                   skeleton_info,
                   dataset,
                   img_root,
                   save_path,
                   start_img_id=0,
                   start_ann_id=0):
    """Save annotations in coco-format.

    :param keypoints_all: keypoint annotations.
    :param annotated_all: images annotated or not.
    :param imgs_all: the array of images.
    :param keypoints_info: infomation about keypoint name.
    :param skeleton_info: infomation about skeleton connection.
    :param dataset: infomation about dataset name.
    :param img_root: the path to save images.
    :param save_path: the path to save transformed annotation file.
    :param start_img_id: the starting point to count the image id.
    :param start_ann_id: the starting point to count the annotation id.
    """
    images = []
    annotations = []

    img_id = start_img_id
    ann_id = start_ann_id

    num_annotations, keypoints_num, _ = keypoints_all.shape

    for i in range(num_annotations):
        img = imgs_all[i]
        keypoints = np.concatenate(
            [keypoints_all[i], annotated_all[i][:, None] * 2], axis=1)

        min_x, min_y = np.min(keypoints[keypoints[:, 2] > 0, :2], axis=0)
        max_x, max_y = np.max(keypoints[keypoints[:, 2] > 0, :2], axis=0)

        anno = {}
        anno['keypoints'] = keypoints.reshape(-1).tolist()
        anno['image_id'] = img_id
        anno['id'] = ann_id
        anno['num_keypoints'] = int(sum(keypoints[:, 2] > 0))
        anno['bbox'] = [
            float(min_x),
            float(min_y),
            float(max_x - min_x + 1),
            float(max_y - min_y + 1)
        ]
        anno['iscrowd'] = 0
        anno['area'] = anno['bbox'][2] * anno['bbox'][3]
        anno['category_id'] = 1

        annotations.append(anno)
        ann_id += 1

        image = {}
        image['id'] = img_id
        image['file_name'] = f'{img_id}.jpg'
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]

        images.append(image)
        img_id += 1

        cv2.imwrite(os.path.join(img_root, image['file_name']), img)

    skeleton = np.concatenate(
        [np.arange(keypoints_num)[:, None], skeleton_info[:, 0][:, None]],
        axis=1) + 1
    skeleton = skeleton[skeleton.min(axis=1) > 0]

    cocotype = {}

    cocotype['info'] = {}
    cocotype['info'][
        'description'] = 'DeepPoseKit-Data Generated by MMPose Team'
    cocotype['info']['version'] = '1.0'
    cocotype['info']['year'] = time.strftime('%Y', time.localtime())
    cocotype['info']['date_created'] = time.strftime('%Y/%m/%d',
                                                     time.localtime())

    cocotype['images'] = images
    cocotype['annotations'] = annotations
    cocotype['categories'] = [{
        'supercategory': 'animal',
        'id': 1,
        'name': dataset,
        'keypoints': keypoints_info,
        'skeleton': skeleton.tolist()
    }]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(cocotype, open(save_path, 'w'), indent=4)
    print('number of images:', img_id)
    print('number of annotations:', ann_id)
    print(f'done {save_path}')


for dataset in ['fly', 'locust', 'zebra']:
    keypoints_info = []
    if dataset == 'fly':
        keypoints_info = [
            'head', 'eyeL', 'eyeR', 'neck', 'thorax', 'abdomen', 'forelegR1',
            'forelegR2', 'forelegR3', 'forelegR4', 'midlegR1', 'midlegR2',
            'midlegR3', 'midlegR4', 'hindlegR1', 'hindlegR2', 'hindlegR3',
            'hindlegR4', 'forelegL1', 'forelegL2', 'forelegL3', 'forelegL4',
            'midlegL1', 'midlegL2', 'midlegL3', 'midlegL4', 'hindlegL1',
            'hindlegL2', 'hindlegL3', 'hindlegL4', 'wingL', 'wingR'
        ]
    elif dataset == 'locust':
        keypoints_info = [
            'head', 'neck', 'thorax', 'abdomen1', 'abdomen2', 'anttipL',
            'antbaseL', 'eyeL', 'forelegL1', 'forelegL2', 'forelegL3',
            'forelegL4', 'midlegL1', 'midlegL2', 'midlegL3', 'midlegL4',
            'hindlegL1', 'hindlegL2', 'hindlegL3', 'hindlegL4', 'anttipR',
            'antbaseR', 'eyeR', 'forelegR1', 'forelegR2', 'forelegR3',
            'forelegR4', 'midlegR1', 'midlegR2', 'midlegR3', 'midlegR4',
            'hindlegR1', 'hindlegR2', 'hindlegR3', 'hindlegR4'
        ]
    elif dataset == 'zebra':
        keypoints_info = [
            'snout', 'head', 'neck', 'forelegL1', 'forelegR1', 'hindlegL1',
            'hindlegR1', 'tailbase', 'tailtip'
        ]
    else:
        NotImplementedError()

    dataset_dir = f'data/DeepPoseKit-Data/datasets/{dataset}'

    with h5py.File(
            os.path.join(dataset_dir, 'annotation_data_release.h5'), 'r') as f:
        # List all groups
        annotations = np.array(f['annotations'])
        annotated = np.array(f['annotated'])
        images = np.array(f['images'])
        skeleton_info = np.array(f['skeleton'])

        annotation_num, kpt_num, _ = annotations.shape

        data_list = np.arange(0, annotation_num)
        np.random.shuffle(data_list)

        val_data_num = annotation_num // 10
        train_data_num = annotation_num - val_data_num

        train_list = data_list[0:train_data_num]
        val_list = data_list[train_data_num:]

        img_root = os.path.join(dataset_dir, 'images')
        os.makedirs(img_root, exist_ok=True)

        save_coco_anno(
            annotations[train_list], annotated[train_list], images[train_list],
            keypoints_info, skeleton_info, dataset, img_root,
            os.path.join(dataset_dir, 'annotations', f'{dataset}_train.json'))
        save_coco_anno(
            annotations[val_list],
            annotated[val_list],
            images[val_list],
            keypoints_info,
            skeleton_info,
            dataset,
            img_root,
            os.path.join(dataset_dir, 'annotations', f'{dataset}_test.json'),
            start_img_id=train_data_num,
            start_ann_id=train_data_num)
