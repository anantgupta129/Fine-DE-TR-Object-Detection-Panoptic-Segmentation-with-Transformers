
import argparse
import json
import os
import time
from io import BytesIO

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from panopticapi.utils import rgb2id
from PIL import Image

from models import build_model

np.random.seed(111)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=25, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--data_path', default='data.test.json', type=str)
    parser.add_argument('--data_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save the results, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--thresh', default=0.85, type=float)
    parser.add_argument('--save_location', type=str,
                        help="the loacation to save to results of mdodel")
    parser.add_argument('--num_test', type=int,
                        help="number of images to be tested")
    return parser


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b


transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@torch.no_grad()
def infer(orig_image, model, postprocessors, classes, device, meta, colors):
    model.eval()
    w, h = orig_image.size
    image = transform(orig_image).unsqueeze(0).to(device)

    out = model(image)

    out["pred_logits"] = out["pred_logits"].cpu()
    out["pred_boxes"] = out["pred_boxes"].cpu()

    probas = out['pred_logits'].softmax(-1)[0, :, :-1]
    # keep = probas.max(-1).values > 0.85
    keep = probas.max(-1).values > args.thresh
    probas = probas[keep].cpu().data.numpy()

    bboxes_scaled = rescale_bboxes(out['pred_boxes'][0, keep], orig_image.size)

    result = postprocessors['panoptic'](out, torch.as_tensor(image.shape[-2:]).unsqueeze(0))[0]
    segments_info = result["segments_info"]
    # Panoptic predictions are stored in a special format png
    panoptic_seg = Image.open(BytesIO(result['png_string']))
    final_w, final_h = panoptic_seg.size
    # We convert the png into an segment id map
    panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
    panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))

    if len(bboxes_scaled) == 0:
        return None

    # plotting boxes
    img_obj = np.array(orig_image)
    for p, box in zip(probas, bboxes_scaled):
        bbox = box.cpu().data.numpy()
        bbox = bbox.astype(np.int32)
        x, y = bbox[0], bbox[1]
        bbox = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ])
        cl = p.argmax()
        c = tuple(map(int, colors[cl]))
        text = f"{classes[cl]}: {p[cl]:.2f}"
        cv2.putText(img_obj, text, (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        bbox = bbox.reshape((4, 2))
        cv2.polylines(img_obj, [bbox], True, c, 2)

    # plotting segmentation
    for i in range(len(segments_info)):
        c = segments_info[i]["category_id"]
        segments_info[i]["category_id"] = \
            meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

    # Finally visualize the prediction
    v = Visualizer(np.array(orig_image.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
    v._default_font_size = 18
    v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0, alpha=0.4)
    img_pan = cv2.resize(v.get_image(), (w, h))

    return (img_obj, img_pan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    # loading model
    model, _, postprocessors = build_model(args)
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model.detr.load_state_dict(checkpoint['model'])

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)

    data = json.load(open("data/test_panoptic.json"))
    im_info = data['images']
    id_to_class = {}
    for cat in data['categories']:
        id_to_class[cat['id']] = cat['name']

    colors = np.random.randint(0, 255, size=(len(id_to_class), 3), dtype=np.int)

    # generating META for custom dataset for visualiztion with detectron
    thing_classes = []
    stuff_classes = []
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}
    thing_count = 0
    stuff_count = 0
    for cat in data['categories']:
        if cat['isthing']:
            thing_classes.append(cat['name'])
            thing_dataset_id_to_contiguous_id[cat['id']] = thing_count
            thing_count += 1
        else:
            stuff_classes.append(cat['name'])
            stuff_dataset_id_to_contiguous_id[cat['id']] = stuff_count
            stuff_count += 1

    DatasetCatalog.register("construction_data", lambda: data['annotations'])
    MetadataCatalog.get("construction_data").set(thing_classes=thing_classes, stuff_classes=stuff_classes,
                                                 thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
                                                 stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id)
    construction_meta = MetadataCatalog.get("construction_data")

    if not os.path.exists(args.save_location):
        os.makedirs(args.save_location)

    i, k = 0, 0
    while True:
        im_path = im_info[k]['file_name']
        im_org = Image.open(im_path)

        start_t = time.perf_counter()
        preds = infer(im_org, model, postprocessors, id_to_class, device, construction_meta, colors)
        end_t = time.perf_counter()

        k += 1

        if preds is not None:
            print("Processed...{} ({:.3f}s) Done:{}..".format(im_path, end_t - start_t, i))

            obj_pred, pan_pred = preds
            im_org = cv2.copyMakeBorder(np.float32(im_org), top=15, bottom=15, left=15, right=15,
                                        borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            obj_pred = cv2.copyMakeBorder(np.float32(obj_pred), top=15, bottom=15, left=15, right=15,
                                          borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            pan_pred = cv2.copyMakeBorder(np.float32(pan_pred), top=15, bottom=15, left=15, right=15,
                                          borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

            cv2.imwrite(f"{args.save_location}/pred_{i}.jpeg", cv2.hconcat([im_org, obj_pred, pan_pred]))
            i += 1

        if i == args.num_test:
            break
