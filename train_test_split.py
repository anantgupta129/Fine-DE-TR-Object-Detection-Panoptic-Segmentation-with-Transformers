import json
import os


train = {
    "info":{
        "description": "Construction Material Panoptic Segmentation & Object Detection Data", 
        "url": "", 
        "version": "1.0", 
        "year": 2021, 
        "contributor": "https://theschoolof.ai/ and COCO val 2017 merged", 
        "date_created": "AUG 2021"
    },
    "licenses": [{"name": "", "id": 0, "url": ""}],
    "images": [],
    "annotations": [],
    "categories": []
}
test = {
    "info":{
        "description": "Construction Material Panoptic Segmentation & Object Detection Data", 
        "url": "", 
        "version": "1.0", 
        "year": 2021, 
        "contributor": "https://theschoolof.ai/ and COCO val 2017 merged", 
        "date_created": "AUG 2021"
    },
    "licenses": [{"name": "", "id": 0, "url": ""}],
    "images": [],
    "annotations": [],
    "categories": []
}

coco = json.load(open("data/coco_val2017/panoptic_val2017.json"))
coco_id_to_cat = {}
for cat in coco['categories']:
    coco_id_to_cat[cat['id']] = cat['name']

coco_cat_to_id = {}
up_coco = json.load(open("data/coco_val2017/coco.json"))
for cat in up_coco['categories']:
    coco_cat_to_id[cat['name']] = cat['id'] 


def create_train_test(test_size=0.05): #20%
    
    test_size *= 100
    categories = os.listdir("data")
    categories.remove("coco_val2017")

    count = 0
    cat_count = 0
    train_im_id = 0
    train_anno_id = 0
    test_im_id = 0
    test_anno_id = 0
    for category in categories:
        data = json.load(open(f"data/{category}/updated_coco.json"))
        k = 0
        images = data['images']

        train['categories'].append(data['categories'][0])
        test['categories'].append(data['categories'][0])
        for im_info in images:
            if count%test_size!=0:
                im_id = im_info['id']
                while True:
                    if k<len(data['annotations']) and data['annotations'][k]['image_id']==im_id:

                        data['annotations'][k]['image_id'] = train_im_id
                        data['annotations'][k]['id'] = train_anno_id
                        
                        train["annotations"].append(
                            data['annotations'][k]
                        )
                        k += 1
                        train_anno_id += 1
                    else:
                        break
                
                im_info['id'] = train_im_id
                train['images'].append(im_info)
                train_im_id += 1
            
            else:
                im_id = im_info['id']
                while True:
                    if k<len(data['annotations']) and data['annotations'][k]['image_id']==im_id:

                        data['annotations'][k]['image_id'] = test_im_id
                        data['annotations'][k]['id'] = test_anno_id
                        test["annotations"].append(
                            data['annotations'][k]
                        )
                        k += 1
                        test_anno_id += 1
                    else:
                        break
                
                im_info['id'] = test_im_id
                test['images'].append(im_info)
                test_im_id += 1
            count += 1
        cat_count += 1

    for cat in train['annotations']:
        category_id = cat['category_id']
        if type(category_id)==str:
            if 'miscellaneous' in category_id:
                cat['category_id'] = cat_count + coco_cat_to_id["miscellaneous stuff"]
            else:
                cat['category_id'] = cat_count + coco_cat_to_id[coco_id_to_cat[int(category_id.split(':')[-1])]]

    for cat in test['annotations']:
        category_id = cat['category_id']
        if type(category_id)==str:
            if 'miscellaneous' in category_id:
                cat['category_id'] = cat_count + coco_cat_to_id["miscellaneous stuff"]
            else:
                cat['category_id'] = cat_count + coco_cat_to_id[coco_id_to_cat[int(category_id.split(':')[-1])]]

    for category in up_coco['categories']:
        category['id'] += cat_count

        train['categories'].append(category)
        test['categories'].append(category)

    for im_info in up_coco['images']: 
        if count%test_size!=0:

            for anno in up_coco['annotations']:
                if im_info['id']==anno['image_id']:
                    for segment in anno['segments_info']:
                        segment['image_id'] = anno['image_id'] + train_im_id
                        segment['category_id'] += cat_count
                        segment['id'] += count
                        train['annotations'].append(segment)
                    break
            
            im_info['id'] += train_im_id
            train['images'].append(im_info)
        else:
            for anno in up_coco['annotations']:
                if im_info['id']==anno['image_id']:
                    for segment in anno['segments_info']:
                        segment['image_id'] = anno['image_id'] + test_im_id
                        segment['category_id'] += cat_count
                        segment['id'] += count
                        test['annotations'].append(segment)
                    break
            
            im_info['id'] += test_im_id
            test['images'].append(im_info)
        count += 1


if __name__=='__main__':
    create_train_test()

    with open(f"data/train.json", "w") as f:
        f.write(json.dumps(train))

    with open(f"data/test.json", "w") as f:
        f.write(json.dumps(test))


