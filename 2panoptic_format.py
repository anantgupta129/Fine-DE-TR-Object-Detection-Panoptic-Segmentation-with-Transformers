import json
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as coco_mask


base_path = 'detr/data'
pan_coco = json.load(open(Path(base_path) / 'coco_val2017' / 'panoptic_val2017.json'))

def convert_coco_poly_to_mask(segmentations, height, width):
    # this function is from facebookresearch_detr_master/datasets/coco.py
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


id2cat = {}
for cat in pan_coco['categories']:
    id2cat[cat['id']] = cat['name']


for phase in ['train.json', 'test.json']:

    data = json.load(open(Path(base_path) / phase))
    cat2id = {}
    for cat in data['categories']:
        cat2id[cat['name']] = cat['id'] 

    pan = {
        'info': data['info'],
        'licenses': data['licenses'],
        'images': data['images'],
        'annotations': [],
        'categories': data['categories']
    }
    k = len(data['annotations'])
    idx = 0

    while idx<k:
        ann = data['annotations'][idx]
        image_id = ann['image_id']
        for image in data['images']:
            if image['id']==image_id:
                file_name = image['file_name']
                height, width = image['height'], image['width']
                break

        ann_info = {
            'segments_info': [],
            'file_name': file_name,
            'image_id': image_id
        }
        flag = False
        while True:
            ann = data['annotations'][idx]
            if 'coco' in file_name and not flag: 
                for annotation in pan_coco['annotations']:
                    if file_name.split('/')[-1].split('.')[0]==annotation['file_name'].split('.')[0]:

                        flag = True
                        ann_info['segments_info'] = annotation['segments_info']

                        for info in ann_info['segments_info']:
                            if id2cat[info['category_id']] in cat2id.keys():
                                info['category_id'] = cat2id[id2cat[info['category_id']]]
                            else:
                                info['category_id'] = cat2id['miscellaneous stuff']
                        break
            else:
                if 'segmentation' in ann.keys() and ann['segmentation']:
                    mask = convert_coco_poly_to_mask([ann['segmentation']], height, width).squeeze(0)
                    segment_info = {
                        'id': ann['id'], 
                        'category_id': ann['category_id'],
                        'segmentation': ann['segmentation'],
                        'iscrowd': ann['iscrowd'], 
                        'bbox': ann['bbox'], 
                        'area': int(torch.sum(mask).item())
                    }
                    ann_info['segments_info'].append(segment_info)
            
            if idx<k-1 and ann['image_id']==data['annotations'][idx+1]['image_id']:
                idx += 1
            else:
                idx += 1
                break
        print(f"{k}, {image_id}, {idx}", end='\r')
            
        pan['annotations'].append(ann_info)

    with open(f"{base_path}/{phase.split('.')[0]}_panoptic.json", "w") as f:
        f.write(json.dumps(pan))
    
    print("\npassed phase:", phase)


