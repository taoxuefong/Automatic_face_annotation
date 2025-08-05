import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

device=None
model=None
stride=None
pt=None
imgsz=[640]
dnn=False
data=ROOT / 'data/coco.yaml'
names=None
half=False
save_dir=None
save_txt=False
augment=False
conf_thres=0.25  # confidence threshold
iou_thres=0.45
max_det=1000
classes=0,  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False
source='./test.jpg'

@smart_inference_mode()
def load():
    global device,model,pt,imgsz,names,stride,save_dir,data
    
    
    # Directories
    save_dir = increment_path(Path(ROOT / 'runs/detect') / 'exp', exist_ok=False)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Load model
    imgsz *= 2 if len(imgsz) == 1 else 1
    device = select_device(device)
    model = DetectMultiBackend('./yolov9.onnx', device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
 
def inference(im0s):
    global device,model,stride, pt,imgsz,names
    im = letterbox(im0s, imgsz, stride=stride, auto=pt)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for now in range(1):
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        
        # Inference
        with dt[1]:
    
            pred = model(im, augment=augment, visualize=False)
            #print(len(pred[0]),len(pred),pred[0][0].shape)
            print(len(pred),pred[0].shape)
            

        # NMS
        with dt[2]:
            pred=[pred,pred]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        result=[]
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
        
            p, im0 = source, im0s.copy()
            
            #p = Path(p)  # to Path
            '''
            save_path = './test1.jpg'  # im.jpg
            annotator = Annotator(im0, line_width=3, example=str(names))
            '''
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    #print(xyxy[0],xyxy[1],xyxy[2],xyxy[3])
                    xyxy_list=[int(t) for t in xyxy]
                    result.append(xyxy_list)
                    #print(xyxy_list)
                    #print(xyxy.tolist())
                    '''
                    label = None if False else (names[c] if False else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    '''
            
            # Save results (image with detections)

            #cv2.imwrite(save_path, im0)
            
    
    # Print results
    
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    return result


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov9.onnx', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='./test.jpg', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #print_args(vars(opt))
    return opt


def init():
    load()
    #opt = parse_opt()
    #check_requirements(exclude=('tensorboard', 'thop'))
    #run(**vars(opt))



if __name__ == "__main__":
    '''
    init()
    im0s = cv2.imread('./test.jpg')  # BGR
    print(inference(im0s))
    '''
