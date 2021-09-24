"""Our task is to train a model for Construction Material Panoptic Segmentation & Object Detection Data
So the rest of the classes in the dataset like car, aeroplane, kite are also Stuff for us
Hence, we are updating default coco dataset 
"""

import json
import os


path_to_lab = r'data\coco_val2017\panoptic_val2017.json'
coco = json.load(open(path_to_lab))
base_path = "data/coco_val2017/images"

things = []
updated_categories = []
id_2_category = {}
category_2_id = {}
category_id = 0

for category in coco['categories']:
    if category["isthing"]:
        things.append(category["id"])
    else:
        id_2_category[category["id"]] = category["name"]
        category["id"] = category_id
        category_2_id[category["name"]] = category_id
        updated_categories.append(category)
        category_id += 1

updated_categories.append(
    {    
        "supercategory": "",
        "isthing": 0,
        "id": category_id,
        "name": "miscellaneous stuff"
    }
)
miscellaneous_id = category_id+1

updated_images = [] 
updated_anno = []
for i in range(1000):
    image = coco['images'][i]
    image_id = image["id"]
    image["file_name"] = "{}/{}".format(base_path, image["file_name"])

    for lab in coco['annotations']:
        if image_id==lab['image_id']:
            segments_info = lab["segments_info"]
            for info in segments_info:
                if info["category_id"] in things:
                    info["category_id"] = miscellaneous_id
                else:
                    info["category_id"] = category_2_id[id_2_category[info["category_id"]]]
            lab['file_name'] = "{}/{}".format(base_path, lab["file_name"])

            updated_images.append(image)
            updated_anno.append(lab)
            break
    

res_file = {
    "info": coco["info"],
    "licenses": coco["licenses"],
    "images": updated_images,
    "annotations": updated_anno,
    "categories": updated_categories
}

with open("data/coco_val2017/coco.json", "w") as f:
    f.write(json.dumps(res_file))


