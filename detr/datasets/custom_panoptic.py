# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from panopticapi.utils import rgb2id
from util.box_ops import masks_to_boxes

from .coco import make_coco_transforms, convert_coco_poly_to_mask


class CustomPanoptic:
    def __init__(self, img_folder, ann_folder, ann_file, transforms=None, return_masks=True):
        with open(ann_file, 'r') as f:
            self.custom = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.custom['images'] = sorted(self.custom['images'], key=lambda x: x['id'])
        # sanity check
        # if "annotations" in self.custom:
        #     for img, ann in zip(self.custom['images'], self.custom['annotations']):
        #         assert img['file_name'][:-4] == ann['file_name'][:-4]

        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self.transforms = transforms
        self.return_masks = return_masks

    def __getitem__(self, idx):
        ann_info = self.custom['annotations'][idx]
        if 'coco' in ann_info['file_name']:
            img_path = ann_info['file_name']

            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            masks, labels = self.get_coco_masks(ann_info)
        else:
            img_path = ann_info['file_name']

            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            masks, labels = self.get_custom_masks(ann_info, h, w)

        target = {}
        target['image_id'] = torch.tensor([ann_info['image_id'] if "image_id" in ann_info else ann_info["id"]])
        if self.return_masks:
            target['masks'] = masks
        target['labels'] = labels

        target["boxes"] = masks_to_boxes(masks)
        # target["boxes"] = torch.tensor([ann['bbox'] for ann in ann_info['segments_info']], dtype=torch.float)

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        if "segments_info" in ann_info:
            for name in ['iscrowd', 'area']:
                target[name] = torch.tensor([ann[name] for ann in ann_info['segments_info']])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.custom['images'])

    def get_height_and_width(self, idx):
        img_info = self.custom['images'][idx]
        height = img_info['height']
        width = img_info['width']
        return height, width

    def get_coco_masks(self, ann_info):
        ann_path = Path(self.ann_folder) / ann_info['file_name'].split('/')[-1].replace('.jpg', '.png')

        if "segments_info" in ann_info:
            masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
            masks = rgb2id(masks)

            ids = np.array([ann['id'] for ann in ann_info['segments_info']])
            masks = masks == ids[:, None, None]

            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.tensor([ann['category_id'] for ann in ann_info['segments_info']], dtype=torch.int64)

        return masks, labels

    def get_custom_masks(self, ann_info, h, w):
        masks = []
        for seg in ann_info['segments_info']:
            masks.append(convert_coco_poly_to_mask([seg['segmentation']], h, w))
        masks = torch.cat(masks, dim=0)
        labels = torch.tensor([ann['category_id'] for ann in ann_info['segments_info']], dtype=torch.int64)

        return masks, labels


def build(image_set, args):
    img_folder_root = Path(args.coco_path)
    ann_folder_root = Path(args.coco_panoptic_path)
    assert img_folder_root.exists(), f'provided custom path {img_folder_root} does not exist'
    assert ann_folder_root.exists(), f'provided custom path {ann_folder_root} does not exist'
    # mode = 'panoptic'
    PATHS = {
        "train": ("", Path("") / 'train_panoptic.json'),
        "val": ("", Path("") / 'test_panoptic.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder_path = img_folder_root / img_folder
    ann_folder = ann_folder_root / "coco_val2017/annotations/panoptic_val2017"
    ann_file = ann_folder_root / ann_file

    dataset = CustomPanoptic(img_folder_path, ann_folder, ann_file,
                             transforms=make_coco_transforms(image_set), return_masks=args.masks)

    return dataset
