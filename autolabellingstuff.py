# !pip install git+https://github.com/cocodataset/panopticapi.git
# !pip install imantics
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

import io
import itertools
import json
import os
import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from imantics import Mask
from panopticapi.utils import rgb2id
from PIL import Image
from pycocotools import mask as coco_mask
from seaborn import color_palette

# from detectron2.detectron2.config import get_cfg
# from detectron2.detectron2.utils.visualizer import Visualizer
# from detectron2.detectron2.data import MetadataCatalog
# from IPython.display import set_matplotlib_formats
# %config InlineBackend.figure_format = 'retina'
# set_matplotlib_formats('retina')
torch.set_grad_enabled(False)
warnings.filterwarnings("ignore", category=UserWarning)
palette = itertools.cycle(color_palette())

# standard PyTorch mean-std input image normalization
transform = T.Compose(
    [
        T.Resize(600),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Selected Device: ", device)

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
model, postprocessor = torch.hub.load(
    "facebookresearch/detr",
    "detr_resnet101_panoptic",
    pretrained=True,
    return_postprocessor=True,
    num_classes=250,
)
model.to(device)
model.eval()


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


def detect_masks(im):
    im_w, im_h = im.size
    img = transform(im).unsqueeze(0)
    out = model(img.to(device))

    # scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
    # threshold the confidence
    # keep = scores > 0.85
    result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

    # The segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
    panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()
    # We retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb2id(panoptic_seg)
    # Finally we color each mask individually
    masks = []
    for id in range(panoptic_seg_id.max() + 1):
        panoptic_seg = np.zeros(list(panoptic_seg_id.shape) + [3])
        panoptic_seg[panoptic_seg_id == id] = np.asarray(next(palette)) * 255
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        panoptic_seg = cv2.cvtColor(panoptic_seg, cv2.COLOR_BGR2GRAY)
        panoptic_seg = cv2.resize(panoptic_seg, (im_w, im_h))
        masks.append(panoptic_seg)

    return masks, result


def remove_overlap(mask, lab_mask):
    row, col = mask.shape
    for i in range(row):
        for j in range(col):
            if mask[i, j] != 0 and lab_mask[i, j] != 0:
                mask[i, j] = 0
    return mask


def get_bbox_segmask(mask, h, w):
    polygons = Mask(mask).polygons()
    segment = []
    for i in polygons.segmentation:
        if len(i) > 20:
            segment.append(i)
    if not segment:
        return [], []

    mask = convert_coco_poly_to_mask([segment], h, w).squeeze()

    bboxes = []
    contours, _ = cv2.findContours(mask.numpy(), 1, 2)
    for cnt in contours:
        # M = cv2.moments(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))

    # combining all the bboxes to form a big bbox
    if len(bboxes) > 0:
        x_min, y_min, w_max, h_max = np.inf, np.inf, 0, 0
        for x, y, w, h in bboxes:
            if x_min > x:
                x_min = x
            if y_min > y:
                y_min = y
            if w_max < x + w:
                w_max = x + w
            if h_max < y + h:
                h_max = y + h

        bbox = [x_min, y_min, w_max - x_min, h_max - y_min]
    else:
        bbox = bboxes[0]
    return bbox, segment


if __name__ == "__main__":

    folders = os.listdir("data")
    folders.remove("coco_val2017")
    im_id = 0
    anno_id = 0
    category_id = 0

    for category in folders:
        if os.path.isfile(f"data/{category}/updated_coco.json"):
            print(f"Skipping {category} as file already updated")
            category_id += 1
            continue
        print(f"Reading From {category}")
        data = json.load(open(f"data/{category}/coco.json"))
        k = 0
        images_info = data["images"]

        res_file = {
            "info": {
                "description": "Construction Material Panoptic Segmentation & Object Detection Data",
                "url": "",
                "version": "1.0",
                "year": 2021,
                "contributor": "https://theschoolof.ai/",
                "date_created": "AUG 2021",
            },
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "supercategory": "construction material",
                    "isthing": 1,
                    "id": category_id,
                    "name": category,
                }
            ],
        }
        for i in range(len(images_info)):
            image = images_info[i]
            im_name = image["file_name"]
            image_id = image["id"]
            h, w = image["height"], image["width"]

            im_path = os.path.join(f"data/{category}/images", im_name)
            if not os.path.isfile(im_path):
                continue

            im = Image.open(im_path).convert("RGB")

            if im.size != (w, h):
                continue
            if k == len(
                data["annotations"]
            ):  # this cond'n is true then are more images then annotions
                break

            masks, result = detect_masks(im)
            print(f"Image {im_id}", end="\r")
            lab_mask_corr = []
            flag = False
            while True:
                if (
                    k < len(data["annotations"])
                    and data["annotations"][k]["image_id"] == image_id
                ):
                    flag = True
                    lab_mask_corr.extend(data["annotations"][k]["segmentation"])
                    res_file["annotations"].append(
                        {
                            "id": anno_id,
                            "image_id": im_id,
                            "category_id": category_id,
                            "segmentation": data["annotations"][k]["segmentation"],
                            "area": data["annotations"][k]["area"],
                            "bbox": data["annotations"][k]["bbox"],
                            "iscrowd": 0,
                            "attributes": data["annotations"][k]["attributes"],
                        }
                    )
                    k += 1
                    anno_id += 1
                else:
                    break
            # checking lab_mask_corr, for some images no anotations
            if flag:
                res_file["images"].append(
                    {
                        "id": im_id,
                        "file_name": im_path,
                        "height": h,
                        "width": w,
                        "license": 0,
                    }
                )
            if (
                k < len(data["annotations"])
                and result["segments_info"]
                and lab_mask_corr
            ):

                lab_mask = (
                    convert_coco_poly_to_mask([lab_mask_corr], h, w).squeeze().numpy()
                )
                for i in range(len(masks)):
                    mask = masks[i]

                    mask = remove_overlap(mask, lab_mask)
                    bbox, cor_seg = get_bbox_segmask(mask, h, w)
                    if not bbox:
                        continue
                    if result["segments_info"][i]["isthing"]:
                        # here we are manuplating the our labelled data the, these category_id will be assigned when we will combine it with coco_val
                        res_file["annotations"].append(
                            {
                                "id": anno_id,
                                "image_id": im_id,
                                "category_id": "assing_later:miscellaneous",
                                "segmentation": cor_seg,
                                "area": bbox[2] * bbox[3],
                                "bbox": bbox,
                                "iscrowd": 0,
                                "attributes": data["annotations"][k]["attributes"],
                            }
                        )
                        anno_id += 1
                    else:
                        res_file["annotations"].append(
                            {
                                "id": anno_id,
                                "image_id": im_id,
                                "category_id": "assing_later:{}".format(
                                    result["segments_info"][i]["category_id"]
                                ),
                                "segmentation": cor_seg,
                                "area": bbox[2] * bbox[3],
                                "bbox": bbox,
                                "iscrowd": 0,
                                "attributes": data["annotations"][k]["attributes"],
                            }
                        )
                        anno_id += 1

            im_id += 1
        print(f"finished {category}: {category_id}")
        category_id += 1

        with open(f"data/{category}/updated_coco.json", "w") as f:
            f.write(json.dumps(res_file))
