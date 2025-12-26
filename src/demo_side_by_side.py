import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from numpy import random
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box, show_seg_result
from lib.core.function import AverageMeter
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def detect(cfg, opt):

    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger, opt.device)

    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    data_iter = enumerate(dataset)
    for _, (path, img, img_det, vid_cap, shapes) in tqdm(data_iter, total=len(dataset)):
        img_orig = img_det.copy()  # Preserve original frame for visualization
        
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out, ll_seg_out = model(img)
        t2 = time_synchronized()
        # if i == 0:
        #     print(det_out)
        inf_out, _ = det_out
        inf_time.update(t2 - t1, img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4 - t3, img.size(0))
        det = det_pred[0]

        if dataset.mode != 'stream':
            save_path = str(save_dir / Path(path).name)
        else:
            save_path = str(save_dir / 'web.mp4')

        _, _, height, width = img.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

        ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1 / ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), 0, 0, is_demo=True)

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_det.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det, label=label_det_pred, color=colors[int(cls)], line_thickness=2)
        
        # Add metrics to the right side image
        cv2.putText(img_det, f'Objects: {len(det)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_det, f'Inference: {t2 - t1:.3f}s', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Combine images side by side
        if img_orig.shape != img_det.shape:
            img_orig = cv2.resize(img_orig, (img_det.shape[1], img_det.shape[0]))
        combined_img = np.hstack((img_orig, img_det))

        if dataset.mode == 'images':
            cv2.imwrite(save_path, combined_img)

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h, w, _ = combined_img.shape  # Use combined image shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(combined_img)
        
        else:
            cv2.imshow('image', combined_img)
            cv2.waitKey(1)  # 1 millisecond

    print(f'Results saved to {save_dir}')
    print(f'Done. ({time.time() - t0:.3f}s)')
    print(f'inf : ({inf_time.avg:.4f}s/frame)   nms : ({nms_time.avg:.4f}s/frame)')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='YOLOP/weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='YOLOP/videos/videos', help='source')  # file/folder   ex:YOLOP/videos/videos
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='YOLOP/videos/output', help='directory to save results')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg, opt)
