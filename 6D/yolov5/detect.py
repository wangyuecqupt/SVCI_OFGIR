# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
	$ python detect.py --weights yolov5s.pt --source 0                               # webcam
													 img.jpg                         # image
													 vid.mp4                         # video
													 screen                          # screenshot
													 path/                           # directory
													 list.txt                        # list of images
													 list.streams                    # list of streams
													 'path/*.jpg'                    # glob
													 'https://youtu.be/Zgi9g1ksQHc'  # YouTube
													 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
	$ python detect.py --weights yolov5s.pt                 # PyTorch
								 yolov5s.torchscript        # TorchScript
								 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
								 yolov5s_openvino_model     # OpenVINO
								 yolov5s.engine             # TensorRT
								 yolov5s.mlmodel            # CoreML (macOS-only)
								 yolov5s_saved_model        # TensorFlow SavedModel
								 yolov5s.pb                 # TensorFlow GraphDef
								 yolov5s.tflite             # TensorFlow Lite
								 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
								 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, \
	LoadSingleImages
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow,
	# check_requirements,
								  colorstr, cv2,
								  increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer,
								  xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode
import numpy
import copy


@smart_inference_mode()
def run(
		weights=ROOT / 'yolov5s.pt',  # model path or triton URL
		source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
		data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
		imgsz=(640, 640),  # inference size (height, width)
		conf_thres=0.25,  # confidence threshold
		iou_thres=0.45,  # NMS IOU threshold
		max_det=1000,  # maximum detections per image
		device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
		view_img=False,  # show results
		save_txt=False,  # save results to *.txt
		save_conf=False,  # save confidences in --save-txt labels
		save_crop=False,  # save cropped prediction boxes
		nosave=False,  # do not save images/videos
		classes=None,  # filter by class: --class 0, or --class 0 2 3
		agnostic_nms=False,  # class-agnostic NMS
		augment=False,  # augmented inference
		visualize=False,  # visualize features
		update=False,  # update all models
		project=ROOT / 'runs/detect',  # save results to project/name
		name='exp',  # save results to project/name
		exist_ok=False,  # existing project/name ok, do not increment
		line_thickness=3,  # bounding box thickness (pixels)
		hide_labels=False,  # hide labels
		hide_conf=False,  # hide confidences
		half=False,  # use FP16 half-precision inference
		dnn=False,  # use OpenCV DNN for ONNX inference
		vid_stride=1,  # video frame-rate stride

		gpu_id=None,
		pretrained=None,
		pretrained_encoder=None,
		codebook=None, cfg_file=None, meta_file=None, dataset_name=None, depth_name=None, color_name=None, imgdir=None,
		randomize=None, network_name=None,
		background_name=None

):
	is_array = False
	if isinstance(source, numpy.ndarray):
		is_array = True
		inp = copy.deepcopy(source)
	source = str(source)
	save_img = not nosave and not source.endswith('.txt')  # save inference images
	is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
	is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
	webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
	screenshot = source.lower().startswith('screen')
	if is_url and is_file:
		source = check_file(source)  # download

	# Directories
	save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
	(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

	# Load model
	device = select_device(device)
	model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
	# stride, names, pt = model.stride, model.names, model.pt
	stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
	imgsz = check_img_size(imgsz, s=stride)  # check image size

	# Dataloader
	bs = 1  # batch_size
	if webcam:
		view_img = check_imshow(warn=True)
		dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
		bs = len(dataset)
	elif screenshot:
		dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
	else:
		# dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
		if is_array:
			dataset = LoadSingleImages(inp, img_size=imgsz, stride=stride, auto=pt and not jit)
		else:
			dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
	vid_path, vid_writer = [None] * bs, [None] * bs

	# Run inference
	model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
	seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
	for path, im, im0s, vid_cap, s in dataset:
		with dt[0]:
			im = torch.from_numpy(im).to(model.device)
			im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
			im /= 255  # 0 - 255 to 0.0 - 1.0
			if len(im.shape) == 3:
				im = im[None]  # expand for batch dim

		# Inference
		with dt[1]:
			visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
			pred = model(im, augment=augment, visualize=visualize)


		# NMS
		with dt[2]:
			pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
		# Second-stage classifier (optional)
		# pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

		# Process predictions
		for i, det in enumerate(pred):  # per image
			seen += 1
			if webcam:  # batch_size >= 1
				p, im0, frame = path[i], im0s[i].copy(), dataset.count
				s += f'{i}: '
			else:
				p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

			p = Path(p)  # to Path
			save_path = str(save_dir / p.name)  # im.jpg
			txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
			s += '%gx%g ' % im.shape[2:]  # print string
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			imc = im0.copy() if save_crop else im0  # for save_crop
			annotator = Annotator(im0, line_width=line_thickness, example=str(names))
			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

				# Print results
				for c in det[:, 5].unique():
					n = (det[:, 5] == c).sum()  # detections per class
					s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

				# Write results
				for *xyxy, conf, cls in reversed(det):
					c = int(cls)  # integer class
					label = names[c] if hide_conf else f'{names[c]}'
					confidence = float(conf)
					confidence_str = f'{confidence:.2f}'

					if save_img or save_crop or view_img:  # Add bbox to image
						c = int(cls)  # integer class
						label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
						annotator.box_label(xyxy, label, color=colors(c, True))
					if save_crop:
						save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

			# Stream results
			im0 = annotator.result()
			if view_img:
				if platform.system() == 'Linux' and p not in windows:
					windows.append(p)
					cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
					cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
				cv2.imshow(str(p), im0)
				cv2.waitKey(1)  # 1 millisecond

		# Print results
		# t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
		# LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
		if update:
			strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', nargs='+', type=str,
						default='/home/yxh/文档/yolov5/runs/train/squirrel/weights/best.pt',
						help='model path or triton URL')
	parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
	parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
	parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
	parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
	parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
	parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='show results')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
	parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
	parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	parser.add_argument('--visualize', action='store_true', help='visualize features')
	parser.add_argument('--update', action='store_true', help='update all models')
	parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
	parser.add_argument('--name', default='circle', help='save results to project/name')
	parser.add_argument('--exist-ok', default=True, help='existing project/name ok, do not increment')
	parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
	parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
	parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
	parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
	parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
	parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')

	parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
						default=0, type=int)
	parser.add_argument('--pretrained', dest='pretrained',
						help='initialize with pretrained checkpoint',
						default=None, type=str)
	parser.add_argument('--pretrained_encoder', dest='pretrained_encoder',
						help='initialize with pretrained encoder checkpoint',
						default=None, type=str)
	parser.add_argument('--codebook', dest='codebook',
						help='codebook',
						default=None, type=str)
	parser.add_argument('--cfg', dest='cfg_file',
						help='optional config file', default=None, type=str)
	parser.add_argument('--meta', dest='meta_file',
						help='optional metadata file', default=None, type=str)
	parser.add_argument('--dataset', dest='dataset_name',
						help='dataset to train on',
						default='shapenet_scene_train', type=str)
	parser.add_argument('--depth', dest='depth_name',
						help='depth image pattern',
						default='*depth.png', type=str)
	parser.add_argument('--color', dest='color_name',
						help='color image pattern',
						default='*color.png', type=str)
	parser.add_argument('--imgdir', dest='imgdir',
						help='path of the directory with the test images',
						default='data/Images', type=str)
	parser.add_argument('--rand', dest='randomize',
						help='randomize (do not use a fixed seed)',
						action='store_true')
	parser.add_argument('--network', dest='network_name',
						help='name of the network',
						default=None, type=str)
	parser.add_argument('--background', dest='background_name',
						help='name of the background file',
						default=None, type=str)

	opt = parser.parse_args()
	opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
	print_args(vars(opt))
	return opt


def main(opt):
	# check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
	return run(**vars(opt))


def pose_init(img):
	opt = parse_opt()
	opt.source = img
	opt.weights = r'//home/yxh/文档/yolov5/runs/train/exp/weights/best.pt'
	return main(opt)


pose_init('/home/yxh/文档/DeepIM-PyTorch-master/data/demo/000004-color.png')
