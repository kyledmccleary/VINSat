from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/detect/17R_s20_200_small3/weights/last.pt')
    model.to('cuda')
    results = model.train(
        resume=True)

    model = YOLO('yolov8m', task='detect')
    model.to('cuda')
    results = model.train(
        data='17R_s20_200.yaml',
        imgsz=1216,
        epochs=300,
        batch=4,
        name='17R_s20_200_medium',
        degrees= 180.0,
        scale= 0.5,
        perspective= 0.00017,
        translate= 0.15,
        fliplr= 0.50,
        mosaic= 0.5,
        box=6.50223,
        cls=0.65876,
        dfl=1.28939,
        hsv_h=0.00959,
        hsv_s=0.59062,
        hsv_v=0.34087)