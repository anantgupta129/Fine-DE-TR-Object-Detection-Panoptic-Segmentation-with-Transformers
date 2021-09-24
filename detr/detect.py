
import argparse
import json
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets.coco import make_coco_transforms
from models import build_model
from PIL import Image


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

    parser.add_argument('--thresh', default=0.5, type=float)
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


@torch.no_grad()
def infer(orig_image, model, classes, device, colors):
    model.eval()
    w, h = orig_image.size
    dummy_target = {
        "size": torch.as_tensor([int(h), int(w)]),
        "orig_size": torch.as_tensor([int(h), int(w)])
    }
    transform = make_coco_transforms("val")
    image, targets = transform(orig_image, dummy_target)
    image = image.unsqueeze(0)
    image = image.to(device)

    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    hooks = [
        model.backbone[-2].register_forward_hook(
                    lambda self, input, output: conv_features.append(output)

        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                    lambda self, input, output: enc_attn_weights.append(output[1])

        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                    lambda self, input, output: dec_attn_weights.append(output[1])

        ),
    ]
    start_t = time.perf_counter()
    outputs = model(image)
    end_t = time.perf_counter()

    outputs["pred_logits"] = outputs["pred_logits"].cpu()
    outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    # keep = probas.max(-1).values > 0.85
    keep = probas.max(-1).values > args.thresh

    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], orig_image.size)
    probas = probas[keep].cpu().data.numpy()

    for hook in hooks:
        hook.remove()

    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0].cpu()

    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]

    if len(bboxes_scaled)==0:
        return None
    
    img = np.array(orig_image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for p, box in zip(probas, bboxes_scaled):
        bbox = box.cpu().data.numpy()
        bbox = bbox.astype(np.int32)
        x,y = bbox[0], bbox[1] 
        bbox = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ])
        cl = p.argmax()
        c = tuple(map(int, colors[cl]))
        text = f"{classes[cl]}: {p[cl]:.2f}"
        cv2.putText(img, text, (x+2,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        bbox = bbox.reshape((4, 2))
        cv2.polylines(img, [bbox], True, c, 2)

    print("Processing...{} ({:.3f}s)".format(im_path, end_t - start_t))
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)
    data = json.load(open("data/test.json"))
    
    num_test = args.num_test 
    model, _, _ = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)

    id_to_class = {}
    for cat in data['categories']:
        id_to_class[cat['id']] = cat['name']
    
    colors = np.random.randint(0, 255, size=(len(id_to_class), 3), dtype=np.int)

    if not os.path.exists(args.save_location):
        os.makedirs(args.save_location)

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(18.5, 10.5)
    im_info = data['images']
    i = 0
    while True:
        k = np.random.randint(0, len(im_info))

        im_path = im_info[k]['file_name']
        im_org = Image.open(im_path)
        im_pred = infer(im_org, model, id_to_class, device, colors)
        
        if im_pred is not None:
            axs[0].imshow(np.array(im_org))
            axs[0].axis('off')
            axs[0].title.set_text('Original Image')

            axs[1].imshow(im_pred)
            axs[1].axis('off')
            axs[1].title.set_text('Predicted Image')
            fig.tight_layout()
            fig.savefig(f"{args.save_location}/pred_{i}.png")
            i += 1

        if i==num_test-1:
            break


