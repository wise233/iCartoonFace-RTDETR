from ultralytics import YOLO
import yaml

# 1. 加载模型
model = YOLO('/root/autodl-tmp/RT-DETR/runs/train/EMP+BiFPN+EIoU/weights/best.pt')

# 2. 打印模型当前的类别定义
print("模型内置类别:", model.names)
# 检查数据集YAML文件
with open(r'/root/autodl-tmp/RT-DETR/ultralytics/cfg/datasets/bvn.yaml', 'r') as f:
    data = yaml.safe_load(f)

print("类别名称:", data['names'])