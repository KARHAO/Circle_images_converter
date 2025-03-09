# 导入所需的库
from PIL import Image, ImageDraw  # PIL库用于图像处理：创建、编辑和保存图像
import os                         # 用于文件和目录操作：路径处理、文件检查等
import argparse                   # 用于解析命令行参数
import re                         # 用于正则表达式匹配：处理尺寸字符串
import sys                        # 用于系统相关操作：退出程序、设置编码等
import math                       # 用于数学计算：计算图片缩放比例等
import cv2                        # OpenCV库用于人脸检测
import numpy as np               # 用于数组操作：图像数据处理
import tkinter as tk            # 用于创建GUI窗口：文件夹选择对话框
from tkinter import filedialog  # 用于文件对话框
import subprocess              # 用于系统命令操作：打开文件夹等

# 设置控制台输出编码（Windows系统下需要）
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# 人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 定义支持的输入格式及其文件扩展名
INPUT_FORMATS = {
    'JPG': ['.jpg', '.jpeg'],    # JPEG格式
    'PNG': ['.png'],             # PNG格式：支持透明背景
    'BMP': ['.bmp'],             # BMP格式：无损位图格式
    'WEBP': ['.webp'],           # WebP格式：Google开发的现代图像格式
    'GIF': ['.gif'],             # GIF格式：支持动画的格式
    'TIFF': ['.tiff', '.tif'],   # TIFF格式：专业图像格式
    'JFIF': ['.jfif'],           # JFIF格式：JPEG文件交换格式
}

# 定义支持的输出格式及其文件扩展名
OUTPUT_FORMATS = {
    'PNG': '.png',    # PNG格式（推荐：支持透明背景）
    'JPEG': '.jpg',   # JPEG格式
    'BMP': '.bmp',    # BMP格式
    'TIFF': '.tiff',  # TIFF格式
    'WEBP': '.webp',  # WebP格式
    'JFIF': '.jfif',  # JFIF格式
}

def get_supported_input_extensions():
    """
    获取所有支持的输入文件扩展名
    
    Returns:
        list: 支持的文件扩展名列表（小写）
    """
    extensions = []
    for format_exts in INPUT_FORMATS.values():
        extensions.extend(format_exts)
    return [ext.lower() for ext in extensions]

def get_output_extension(format_name):
    """
    获取指定输出格式的文件扩展名
    
    Args:
        format_name (str): 输出格式名称
    
    Returns:
        str: 对应的文件扩展名
        
    Raises:
        ValueError: 如果指定的格式不支持
    """
    format_name = format_name.upper()
    if format_name not in OUTPUT_FORMATS:
        raise ValueError(f'不支持的输出格式: {format_name}')
    return OUTPUT_FORMATS[format_name]

def parse_size(size_str):
    """
    解析尺寸字符串，支持像素(px)、毫米(mm)、厘米(cm)
    
    Args:
        size_str (str): 尺寸字符串，例如: '500px', '50mm', '5cm'
    
    Returns:
        int: 像素值
        
    Raises:
        ValueError: 如果尺寸字符串格式不正确
    """
    # 使用正则表达式匹配数字和单位
    match = re.match(r'^(\d+\.?\d*)(px|mm|cm)$', size_str.lower())
    if not match:
        raise ValueError('尺寸格式不正确。正确格式示例: 500px, 50mm, 5cm')
    
    value = float(match.group(1))
    unit = match.group(2)
    
    # 根据单位转换为像素值（基准DPI为300）
    if unit == 'px':
        return int(value)
    elif unit == 'mm':
        return int(value / 25.4 * 300)  # 1英寸 = 25.4毫米
    elif unit == 'cm':
        return int(value * 10 / 25.4 * 300)  # 1厘米 = 10毫米
    
    raise ValueError('不支持的单位。支持的单位: px, mm, cm')

def get_size_in_unit(pixels, unit, dpi):
    """
    将像素值转换为指定单位的尺寸
    
    Args:
        pixels (int): 像素值
        unit (str): 目标单位（'px', 'mm', 'cm'）
        dpi (int): DPI值
    
    Returns:
        float: 转换后的尺寸值
    """
    if unit == 'px':
        return pixels
    
    inches = pixels / dpi
    if unit == 'mm':
        return inches * 25.4
    elif unit == 'cm':
        return inches * 2.54
    
    return pixels

def calculate_inscribed_square_size(circle_diameter):
    """
    计算内接正方形的边长（正方形在圆内）
    
    Args:
        circle_diameter (int): 圆的直径
    
    Returns:
        int: 内接正方形的边长
    """
    # 内接正方形的边长等于圆直径除以根号2
    return int(circle_diameter / math.sqrt(2))

def calculate_circumscribed_square_size(circle_diameter):
    """
    计算外接正方形的边长（正方形在圆外）
    
    Args:
        circle_diameter (int): 圆的直径
    
    Returns:
        int: 外接正方形的边长
    """
    # 外接正方形的边长等于圆的直径
    return circle_diameter

def detect_faces(image_path):
    """
    检测图像中的人脸，使用多个级联分类器提高准确性
    
    Args:
        image_path (str): 输入图像路径
        
    Returns:
        list: 人脸边界框列表 [(x1, y1, x2, y2), ...]
    """
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f'⚠ 警告: 无法读取图像 {os.path.basename(image_path)}')
            return []
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 直方图均衡化，提高图像质量
        gray = cv2.equalizeHist(gray)
        
        # 加载人脸检测器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        if face_cascade.empty() or profile_cascade.empty():
            print(f'⚠ 警告: 人脸检测器加载失败，将使用图像中心')
            return []
        
        # 检测正面人脸（调整参数提高准确性）
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,  # 稍微增加以平衡速度和准确性
            minNeighbors=4,    # 降低以提高检测率
            minSize=(30, 30),  # 最小人脸大小
            maxSize=(0, 0),    # 不限制最大大小
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # 检测侧面人脸
        profile_faces = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # 检测镜像侧面人脸
        flipped = cv2.flip(gray, 1)
        profile_faces_flipped = profile_cascade.detectMultiScale(
            flipped,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # 合并所有检测结果
        face_boxes = []
        
        # 添加正面人脸
        for (x, y, w, h) in faces:
            face_boxes.append((x, y, x + w, y + h))
        
        # 添加侧面人脸
        for (x, y, w, h) in profile_faces:
            face_boxes.append((x, y, x + w, y + h))
        
        # 添加镜像侧面人脸（需要调整x坐标）
        width = gray.shape[1]
        for (x, y, w, h) in profile_faces_flipped:
            face_boxes.append((width - (x + w), y, width - x, y + h))
        
        # 如果没有检测到人脸，使用更宽松的参数重试
        if not face_boxes:
            faces_retry = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,     # 更宽松的参数
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, w, h) in faces_retry:
                face_boxes.append((x, y, x + w, y + h))
        
        return face_boxes
        
    except Exception as e:
        print(f'⚠ 警告: 人脸检测出错 ({str(e)})，将使用图像中心')
        return []

def select_folder(title):
    """
    弹出文件夹选择对话框
    
    Args:
        title (str): 对话框标题
    
    Returns:
        str: 选择的文件夹路径，如果取消则返回None
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder = filedialog.askdirectory(title=title)
    return folder if folder else None

def open_folder(path):
    """
    打开指定文件夹
    
    Args:
        path (str): 文件夹路径
    """
    try:
        if os.path.exists(path):
            # Windows系统使用explorer打开文件夹
            os.startfile(path)
    except Exception as e:
        print(f'打开文件夹时出错: {str(e)}')

def create_circular_image(input_path, output_path, size=(500, 500), dpi=300, fit_corners=False, input_format='PNG', output_format='PNG', face_center=False):
    """
    创建圆形图片，支持三种模式：
    1. 默认模式：根据长边等比例缩放并相切
    2. 四角相切模式(fit_corners=True)：图片完整显示，四角与圆形相切
    3. 人脸检测模式(face_center=True)：自动检测人脸并居中，图片填满圆形区域
    
    Args:
        input_path (str): 输入图像路径
        output_path (str): 输出图像路径
        size (tuple): 圆形画布大小（宽度，高度）
        dpi (int): 输出图像DPI，影响最终分辨率
        fit_corners (bool): 是否使用四角相切模式
        input_format (str): 输入图像格式
        output_format (str): 输出图像格式
        face_center (bool): 是否启用人脸检测和居中
    """
    # 打开图像
    img = Image.open(input_path)
    
    # 获取画布尺寸
    canvas_size = size[0]  # 使用宽度作为圆形直径
    
    # 根据DPI计算实际输出分辨率
    output_size = int(canvas_size * dpi / 300)  # 基准DPI为300
    
    if face_center:
        # 人脸检测模式：识别人脸并居中，图片填满圆形区域
        face_boxes = detect_faces(input_path)
        if face_boxes and len(face_boxes) > 0:
            # 计算所有人脸的边界框
            min_x = min(box[0] for box in face_boxes)
            min_y = min(box[1] for box in face_boxes)
            max_x = max(box[2] for box in face_boxes)
            max_y = max(box[3] for box in face_boxes)
            
            # 计算人脸区域的中心点
            face_center_x = (min_x + max_x) // 2
            face_center_y = (min_y + max_y) // 2
            
            # 计算人脸区域的宽度和高度
            face_width = max_x - min_x
            face_height = max_y - min_y
            
            # 根据人脸大小动态调整边距
            face_size_ratio = (face_width * face_height) / (img.width * img.height)
            if face_size_ratio < 0.1:  # 小脸
                padding_ratio = 1.2  # 增加小脸的边距
            elif face_size_ratio < 0.3:  # 中等大小的脸
                padding_ratio = 1.0
            else:  # 大脸
                padding_ratio = 0.8  # 减少大脸的边距
            
            # 计算裁剪区域大小，确保包含完整的人脸
            padding = max(face_width, face_height) * padding_ratio
            crop_size = max(face_width + padding * 2, face_height + padding * 2)
            
            # 确保裁剪区域不超出原图范围，同时保持人脸居中
            left = max(0, face_center_x - crop_size // 2)
            top = max(0, face_center_y - crop_size // 2)
            right = min(img.width, face_center_x + crop_size // 2)
            bottom = min(img.height, face_center_y + crop_size // 2)
            
            # 如果裁剪区域超出边界，调整以保持居中
            if right - left < crop_size:
                if left == 0:
                    right = min(img.width, left + crop_size)
                else:
                    left = max(0, right - crop_size)
            if bottom - top < crop_size:
                if top == 0:
                    bottom = min(img.height, top + crop_size)
                else:
                    top = max(0, bottom - crop_size)
            
            # 裁剪图片
            img = img.crop((left, top, right, bottom))
            print(f'✓ 已检测到人脸: {os.path.basename(input_path)}')
        else:
            print(f'⚠ 未检测到人脸: {os.path.basename(input_path)}，使用图像中心')
        
        # 计算缩放比例，确保填满圆形区域
        scale = output_size / min(img.width, img.height)
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        
    elif fit_corners:
        # 四角相切模式：图片完整显示，四角与圆形相切
        # 计算图片初始尺寸，保持原始比例
        img_ratio = img.height / img.width
        
        if img_ratio > 1:  # 竖图
            new_height = output_size
            new_width = int(new_height / img_ratio)
        else:  # 横图或正方形
            new_width = output_size
            new_height = int(new_width * img_ratio)
        
        # 计算对角线长度
        diagonal = math.sqrt(new_width**2 + new_height**2)
        
        # 计算缩放比例，使对角线等于圆的直径
        scale = output_size / diagonal
        new_width = int(new_width * scale)
        new_height = int(new_height * scale)
        
    else:
        # 默认模式：根据长边计算缩放比例
        scale = output_size / max(img.width, img.height)
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
    
    # 缩放图片（使用LANCZOS重采样算法，效果最好）
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 创建透明背景的画布
    canvas = Image.new('RGBA', (output_size, output_size), (0, 0, 0, 0))
    
    # 计算图片在画布上的位置（居中）
    paste_x = (output_size - new_width) // 2
    paste_y = (output_size - new_height) // 2
    
    # 将图片粘贴到画布上
    canvas.paste(img, (paste_x, paste_y))
    
    # 创建圆形遮罩
    mask = Image.new('L', (output_size, output_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, output_size-1, output_size-1), fill=255)
    
    # 应用圆形遮罩
    canvas.putalpha(mask)
    
    # 保存结果，指定DPI
    canvas.save(output_path, format=output_format, dpi=(dpi, dpi))

def process_directory(input_dir, output_dir, size=(500, 500), dpi=300, fit_corners=False, input_format='PNG', output_format='PNG', face_center=False):
    """
    批量处理输入目录中的所有图片
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        size (tuple): 圆形画布大小（宽度，高度）
        dpi (int): 输出图像DPI，影响最终分辨率
        fit_corners (bool): 是否使用四角相切模式
        input_format (str): 输入图像格式
        output_format (str): 输出图像格式
        face_center (bool): 是否启用人脸检测和居中
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据处理模式设置输出文件名前缀
    mode_prefix = ""
    if face_center:
        mode_prefix = "face_"  # 人脸检测模式前缀
    elif fit_corners:
        mode_prefix = "fit_"   # 四角相切模式前缀
    else:
        mode_prefix = "default_"  # 默认模式前缀
    
    # 获取所有文件
    all_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    # 从INPUT_FORMATS字典获取所有支持的文件扩展名
    supported_formats = []
    for extensions in INPUT_FORMATS.values():
        supported_formats.extend(extensions)
    
    # 对文件进行分类（支持的图片和不支持的文件）
    image_files = []
    unsupported_files = []
    for file in all_files:
        if any(file.lower().endswith(ext) for ext in supported_formats):
            image_files.append(file)
        else:
            unsupported_files.append(file)
    
    # 显示文件统计信息
    print(f'\n文件统计:')
    print(f'总文件数: {len(all_files)} 个')
    print(f'支持处理的图片: {len(image_files)} 个')
    if unsupported_files:
        print(f'不支持处理的文件: {len(unsupported_files)} 个')
        for file in unsupported_files:
            print(f'  - {file}')
    print()
    
    if not image_files:
        print('没有找到可处理的图片文件')
        return
    
    # 处理结果统计
    processed_files = []  # 成功处理的文件
    failed_files = []    # 处理失败的文件
    
    # 处理每个图片
    for i, file in enumerate(image_files, 1):
        try:
            # 构建输入输出路径
            input_path = os.path.join(input_dir, file)
            
            # 添加模式前缀到输出文件名
            filename = os.path.splitext(file)[0]
            output_filename = f"{mode_prefix}{filename}.{output_format.lower()}"
            output_path = os.path.join(output_dir, output_filename)
            
            # 创建圆形图片
            create_circular_image(
                input_path=input_path,
                output_path=output_path,
                size=size,
                dpi=dpi,
                fit_corners=fit_corners,
                input_format=input_format,
                output_format=output_format,
                face_center=face_center
            )
            
            # 显示处理进度
            print(f'[{i}/{len(image_files)}] ✓ {file} -> {output_filename}')
            processed_files.append(file)
            
        except Exception as e:
            # 处理失败时记录错误信息
            print(f'[{i}/{len(image_files)}] ✗ {file} - 处理失败: {str(e)}')
            failed_files.append((file, str(e)))
    
    # 显示最终处理结果统计
    print('\n处理结果统计:')
    print(f'输入文件夹总文件数: {len(all_files)} 个')
    print(f'可处理的图片: {len(image_files)} 个')
    print(f'成功处理: {len(processed_files)} 个')
    if failed_files:
        print(f'处理失败: {len(failed_files)} 个')
        print('失败的文件:')
        for file, error in failed_files:
            print(f'  - {file}: {error}')
    if unsupported_files:
        print(f'不支持的文件: {len(unsupported_files)} 个')
    
    print('\n处理完成!')

def main():
    """
    主函数：处理命令行参数并执行图片转换
    """
    # 创建命令行参数解析器，添加详细的程序描述
    parser = argparse.ArgumentParser(
        description='圆形图片转换工具\n'
                  '将各种格式的图片转换为带透明背景的圆形图片\n'
                  '支持三种转换模式：默认模式、四角相切模式和人脸检测模式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
以下是帮助指令和示例:

使用示例:
  # 使用默认设置（500px，PNG格式）
  python circle_image_converter.py
  
  # 指定输出尺寸为1000像素，使用四角相切模式
  python circle_image_converter.py --size 1000px --fit
  
  # 使用人脸检测模式，输出为300DPI的JPEG格式
  python circle_image_converter.py --face --dpi 300 --output-format JPEG
  
支持的输入格式:
  PNG, JPG/JPEG, BMP, GIF, WebP, TIFF, JFIF
  
支持的输出格式:
  PNG（推荐，支持透明背景）, JPEG, BMP, WebP, TIFF, JFIF
  
尺寸单位支持:
  px（像素）, mm（毫米）, cm（厘米）
  示例: 500px, 50mm, 5cm

命令行选项:
--size：设置圆形画布尺寸，支持以下格式：
  - 像素：500px（默认）
  - 毫米：50mm
  - 厘米：5cm
--dpi：设置输出图片DPI，默认300
--fit：启用四角相切模式
--face：启用人脸检测和居中模式
--input-format：指定输入图片格式，默认PNG
--output-format：指定输出图片格式，默认PNG
''')
    
    # 添加命令行参数
    parser.add_argument('--size', default='500px',
                      help='圆形画布尺寸，支持px/mm/cm单位 (默认: 500px)')
    parser.add_argument('--dpi', type=int, default=300,
                      help='输出图片DPI，影响最终分辨率 (默认: 300)')
    parser.add_argument('--fit', action='store_true',
                      help='启用四角相切模式，确保原图完整显示且四角与圆形相切')
    parser.add_argument('--input-format', default='PNG', choices=list(INPUT_FORMATS.keys()),
                      help='输入图片格式 (默认: PNG)')
    parser.add_argument('--output-format', default='PNG', choices=list(OUTPUT_FORMATS.keys()),
                      help='输出图片格式 (默认: PNG)')
    parser.add_argument('--face', action='store_true',
                      help='启用人脸检测模式，自动检测人脸并居中')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        # 选择输入文件夹
        input_dir = select_folder('选择输入文件夹')
        if not input_dir:
            print('未选择输入文件夹，程序退出')
            sys.exit(0)
        
        # 选择输出文件夹
        output_dir = select_folder('选择输出文件夹')
        if not output_dir:
            print('未选择输出文件夹，程序退出')
            sys.exit(0)
        
        # 解析画布尺寸
        canvas_size = parse_size(args.size)
        size = (canvas_size, canvas_size)
        
        # 显示输出设置
        print('\n输出图片设置:')
        print(f'画布尺寸:{get_size_in_unit(canvas_size, "mm", 300):.1f}×{get_size_in_unit(canvas_size, "mm", 300):.1f} 毫米')
        print(f'         {get_size_in_unit(canvas_size, "cm", 300):.1f}×{get_size_in_unit(canvas_size, "cm", 300):.1f} 厘米')
        
        # 计算实际输出分辨率
        output_size = int(canvas_size * args.dpi / 300)
        print(f'实际分辨率: {output_size}×{output_size} 像素 ({args.dpi} DPI)')
        print(f'输入格式: {args.input_format}')
        print(f'输出格式: {args.output_format}')
        print(f'图片缩放: {"四角相切模式" if args.fit else "填充模式"}')
        print(f'人脸检测: {"启用" if args.face else "禁用"}')
        print()
        
        # 处理整个目录
        process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            size=size,
            dpi=args.dpi,
            fit_corners=args.fit,
            input_format=args.input_format,
            output_format=args.output_format,
            face_center=args.face
        )
        
        # 处理完成后打开输出文件夹
        open_folder(output_dir)
        
    except ValueError as e:
        print(f'错误: {str(e)}')
        sys.exit(1)
    except Exception as e:
        print(f'发生错误: {str(e)}')
        sys.exit(1)

# 程序入口点
if __name__ == '__main__':
    main()
