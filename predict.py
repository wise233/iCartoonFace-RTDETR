import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


def predict_img(model_path, image_path, conf_threshold=0.7):
    model = RTDETR(model_path)
    results = model.predict(
        source=image_path, 
        imgsz=640,
        task='detect',
        conf=conf_threshold,
        agnostic_nms=True,
        save=True,
        project='runs/predict',
        name='exp',
        max_det=1000
    )
    for r in results:
        img_height, img_width = r.orig_shape
        for box in r.boxes:
            normalized_coords = box.xyxyn[0].cpu().numpy()
            print("normalized_coords:",normalized_coords)
            pixel_coords = normalized_coords * [img_width, img_height, img_width, img_height]
            x_min, y_min, x_max, y_max = pixel_coords
            print(f"像素坐标: x_min={x_min:.1f}, y_min={y_min:.1f}, x_max={x_max:.1f}, y_max={y_max:.1f}")
        
    # 视频文件路径
    video_path = "your_video.mp4"  # 替换为你的视频路径
def predict_video(model_path, video_path, conf_threshold=0.5):
    model = RTDETR(model_path)
    # 执行预测
    results = model.predict(
        source=video_path,          # 关键修改：指向视频文件
        imgsz=640,
        task='detect',
        conf=conf_threshold,
        agnostic_nms=True,
        save=True,                  # 自动保存检测结果视频
        project='runs/predict',
        name='exp',
        stream=False,               # 对于视频文件，通常设置为False
        # save_txt=True             # 可选：保存检测结果的文本文件
        # save_conf=True           # 可选：在输出中保存置信度
    )
        

if __name__ == '__main__':
    model_path = '/root/autodl-tmp/RT-DETR/runs/train/EMP+BiFPN+EIoU/weights/best.pt' 
    image_path = '/root/autodl-tmp/RT-DETR/1.jpg' 
    video_path = '/root/autodl-tmp/RT-DETR/CartoonFace.mp4'
    predict_img(model_path, image_path, conf_threshold=0.3)
    #predict_video(model_path, video_path, conf_threshold=0.7)