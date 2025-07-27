import os
import csv
from PIL import Image
import time

def convert_to_yolo_format(csv_path, img_dir, output_dir, max_count, is_val=False):
    """转换CSV标注到YOLO格式
    Args:
        csv_path: CSV文件路径
        img_dir: 图片目录
        output_dir: YOLO标注输出目录
        max_count: 最大图片数量
        is_val: 是否为验证集/测试集
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_images = set()
    image_count = 0
    # 第一步：预读取CSV数据到内存
    csv_data = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            img_name = row[0]
            # 添加到图像数据字典
            if img_name not in csv_data:
                csv_data[img_name] = []
            csv_data[img_name].append(row)
    print(f"已加载 {len(csv_data)} 张图片的标注数据")
    # 第二步：处理每张图片
    for img_name, annotations in csv_data.items():
        # 提取图片编号 (00001, 00002等)
        img_num = img_name.split('_')[-1].split('.')[0]
        # 检查图片数量限制
        if is_val:
            if int(img_num) > max_count - 1:  # 验证集从00000开始
                continue
        else:
            if int(img_num) > max_count:  # 训练集从00001开始
                continue
        img_path = os.path.join(img_dir, img_name)
        # 检查图片是否存在
        if not os.path.exists(img_path):
            print(f"警告: 图片 {img_path} 不存在，跳过")
            continue
        # 获取图片尺寸
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"无法打开图片 {img_path}: {e}")
            continue
        # 准备YOLO标注文件路径
        txt_name = img_name.replace('.jpg', '.txt')
        txt_path = os.path.join(output_dir, txt_name)
        # 处理当前图片的所有标注
        with open(txt_path, 'w') as txt_file:
            for row in annotations:
                # 解析坐标 (训练集5列，验证集6列)
                if len(row) == 5:  # 训练集格式: img,x1,y1,x2,y2
                    x_min, y_min, x_max, y_max = map(float, row[1:5])
                else:  # 验证集格式: img,x1,y1,x2,y2,class
                    x_min, y_min, x_max, y_max = map(float, row[1:5])
                # 计算YOLO格式的归一化坐标
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height    
                # 写入YOLO格式行 (类别固定为0)
                txt_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        # 更新处理状态
        image_count += 1
        if image_count % 500 == 0:
            print(f"已处理 {image_count} 张图片...")
    print(f"处理完成! 共转换 {image_count} 张图片的标注数据")
    return image_count
def main():
    # 配置路径 - 请根据实际情况修改这些路径
    train_csv = "/root"
    val_csv = "/root"
    train_img_dir = "/root"
    val_img_dir = "/root"
    yolo_output_dir = "/root"
    # 创建输出目录
    train_output = os.path.join(yolo_output_dir, "train")
    val_output = os.path.join(yolo_output_dir, "val")
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(val_output, exist_ok=True)
    # 转换训练集 (5000张图片)
    print("开始转换训练集...")
    start_time = time.time()
    train_count = convert_to_yolo_format(
        csv_path=train_csv,
        img_dir=train_img_dir,
        output_dir=train_output,
        max_count=5000
    )
    print(f"训练集转换完成! 共转换 {train_count} 张图片，耗时 {time.time()-start_time:.2f} 秒")
    # 转换测试集 (2000张图片)
    print("\n开始转换测试集...")
    start_time = time.time()
    val_count = convert_to_yolo_format(
        csv_path=val_csv,
        img_dir=val_img_dir,
        output_dir=val_output,
        max_count=2000,
        is_val=True
    )
    print(f"测试集转换完成! 共转换 {val_count} 张图片，耗时 {time.time()-start_time:.2f} 秒")
    print("\n所有转换完成!")
    print(f"总转换图片数量: {train_count + val_count} (训练集: {train_count}, 测试集: {val_count})")

if __name__ == "__main__":
    main()