from ultralytics import YOLO
import cv2
import numpy as np
import glob


def collect_datasets():
    source = glob.glob('/home/yxh/文档/ObjectDatasetTools-master/LINEMOD(复件)/arqc/JPEGImages/*.jpg')
    source = sorted(source, key=lambda x: int(x.split('/')[-1].split('.')[0]), reverse=True)
    print(len(source))

    if len(source) > 0:
        import transforms3d as t3d
        pose_gts = glob.glob('/home/yxh/文档/ObjectDatasetTools-master/LINEMOD(复件)/arqc/pose_gts/*.npy')
        pose_gts = sorted(pose_gts, key=lambda x: int(x.split('/')[-1].split('.')[0]), reverse=True)
        gts = []
        for each in pose_gts:
            pose = np.load(each)
            gts.append(np.append(t3d.quaternions.mat2quat(pose[:3, :3]), values=[pose[0, 3], pose[1, 3], pose[2, 3]]))

    return source, gts




def pose_init_v8(init_queue, track_queue, infer_cfg):
    if infer_cfg["track_target"] == "arqc":
        model = YOLO(r'/home/yxh/文档/ultralytics/runs/detect/train/weights/best.pt')
    elif infer_cfg["track_target"] == "squirrel":
        model = YOLO(r'/home/yxh/文档/ultralytics/runs/detect/squirrel/weights/best.pt')
    elif infer_cfg["track_target"] == "can":
        model = YOLO(r'/home/yxh/文档/ultralytics/runs/detect/can/weights/best.pt')

    if infer_cfg["data_mode"] == 'camera':
        # 获取摄像头内容，参数 0 表示使用默认的摄像头
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()  # 读取摄像头的一帧图像
            if success:
                results = model.predict(source=frame, imgsz=(640, 640), conf=0.05)  # 对当前帧进行目标检测并显示结果

            output = np.zeros((640, 640, 3))
            output[80:480 + 80, :, :] = frame
            output = output.astype(np.uint8)

            if len(results[0].boxes.cls) == 0:
                # print('yolo定位失败!')
                init_queue.queue.clear()
                init_queue.put({'im': output, 'cls_box': [int(-1), int(0), int(0), int(640), int(640)], 'gts': None})
            else:
                box = [int(i) for i in results[0].boxes.xyxy[0]]
                cls = int(results[0].boxes.cls[0])
                init_queue.queue.clear()
                init_queue.put({'im': output, 'cls_box': [int(cls), int(box[0]), int(box[1] + 80), int(box[2] - box[0]),
                                                          int(box[3] - box[1])], 'gts': None})

            inferencing = track_queue.get()
            cv2.imshow('inferencing...', inferencing)
            cv2.waitKey(1)
            track_queue.queue.clear()

    elif infer_cfg["data_mode"] == 'images':
        images, pose_gts = collect_datasets()
        for image, pose in zip(images, pose_gts):
            frame = cv2.imread(image)
            input_frame = np.zeros((640, 640, 3))
            input_frame[80:480 + 80, :, :] = frame
            input_frame = input_frame.astype(np.uint8)
            results = model.predict(source=input_frame, imgsz=(640, 640), conf=0.05)  # 对当前帧进行目标检测并显示结果

            if len(results[0].boxes.cls) == 0:
                # print('yolo定位失败!')
                init_queue.queue.clear()
                init_queue.put(
                    {'im': input_frame, 'cls_box': [int(-1), int(0), int(0), int(640), int(640)], 'gts': pose})
            else:
                box = [int(i) for i in results[0].boxes.xyxy[0]]
                cls = int(results[0].boxes.cls[0])
                init_queue.queue.clear()
                init_queue.put({'im': input_frame, 'cls_box': [int(cls), int(box[0]), int(box[1]), int(box[2] - box[0]),
                                                               int(box[3] - box[1])], 'gts': pose})

            inferencing = track_queue.get()
            cv2.imshow('inferencing...', inferencing)
            cv2.waitKey(1)
            track_queue.queue.clear()


def my_train():
    # 加载一个模型
    # model = YOLO('yolov8n.yaml')  # 从YAML建立一个新模型
    # model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 从YAML建立并转移权重

    # 训练模型
    results = model.train(data='coco128.yaml', epochs=100, imgsz=640)


def fps():
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 从YAML建立并转移权重
    cap = cv2.VideoCapture(0)
    import time
    n = 0
    time1 = time.time()
    while n <= 1392:
        success, frame = cap.read()  # 读取摄像头的一帧图像
        if success:
            results = model.predict(source=frame, imgsz=(640, 640), conf=0.05)  # 对当前帧进行目标检测并显示结果

        output = np.zeros((640, 640, 3))
        output[80:480 + 80, :, :] = frame
        output = output.astype(np.uint8)
        n = n + 1
        print(n)

    time2 = time.time()
    print(1392 / (time2 - time1))


if __name__ == '__main__':
    my_train()
