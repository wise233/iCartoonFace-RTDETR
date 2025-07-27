import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('rtdetr-EMO.yaml')
    # model.load('') # loading pretrain weights
    model.train(data=r'/root/autodl-tmp/RT-DETR/ultralytics/cfg/datasets/bvn.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                workers=10,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                # amp=True
                )