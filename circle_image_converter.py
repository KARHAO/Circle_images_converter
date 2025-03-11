# 导入所需的库
from PIL import Image, ImageDraw  # PIL库用于图像处理：创建、编辑和保存图像，支持多种格式和透明度
import os                         # 用于文件和目录操作：路径处理、文件检查、目录创建等
import argparse                   # 用于解析命令行参数，提供友好的命令行界面
import re                         # 用于正则表达式匹配：处理尺寸字符串，支持多种单位
import sys                        # 用于系统相关操作：退出程序、设置编码、错误处理等
import math                       # 用于数学计算：图片缩放、圆形计算等
import cv2                        # OpenCV库用于人脸和特征检测：支持多种特征识别
import numpy as np               # 用于数组操作：图像数据处理、矩阵运算等
import tkinter as tk            # 用于创建GUI窗口：提供文件夹选择对话框
from tkinter import filedialog  # 用于文件对话框：用户友好的文件选择界面
import subprocess              # 用于系统命令操作：打开输出文件夹等

# 设置控制台输出编码（Windows系统下需要）
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# 初始化人脸检测器（使用预训练的Haar级联分类器）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 定义支持的输入格式及其文件扩展名
INPUT_FORMATS = {
    'JPG': ['.jpg', '.jpeg'],    # JPEG格式：常用的有损压缩格式
    'PNG': ['.png'],             # PNG格式：支持透明背景的无损格式
    'BMP': ['.bmp'],             # BMP格式：无损位图格式，文件较大
    'WEBP': ['.webp'],           # WebP格式：Google开发的现代图像格式，支持有损和无损
    'GIF': ['.gif'],             # GIF格式：支持动画的格式，本工具仅处理第一帧
    'TIFF': ['.tiff', '.tif'],   # TIFF格式：专业图像格式，支持高位深
    'JFIF': ['.jfif'],           # JFIF格式：JPEG文件交换格式的变体
}

# 定义支持的输出格式及其文件扩展名
OUTPUT_FORMATS = {
    'PNG': '.png',    # PNG格式（推荐：支持透明背景，无损压缩）
    'JPEG': '.jpg',   # JPEG格式（有损压缩，文件小，不支持透明）
    'BMP': '.bmp',    # BMP格式（无损，文件大，不支持透明）
    'TIFF': '.tiff',  # TIFF格式（专业用途，支持高位深）
    'WEBP': '.webp',  # WebP格式（现代格式，兼具压缩率和质量）
    'JFIF': '.jfif',  # JFIF格式（JPEG变体）
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

def show_processing_image(image, features=None, window_name="处理进度预览", window_size=(800, 600)):
    """
    显示处理过程中的图像预览，包括特征标记和实时更新
    
    显示特点：
    - 固定大小窗口
    - 自动缩放以适应窗口
    - 特征标记使用不同颜色
    - 居中显示
    
    特征标记颜色：
    - 绿色：正面人脸
    - 蓝色：侧面人脸
    - 红色：躯干
    - 青色：眼睛
    
    Args:
        image: OpenCV格式的图像数据
        features (dict): 检测到的特征信息字典
        window_name (str): 预览窗口标题
        window_size (tuple): 窗口大小 (宽, 高)
    """
    try:
        # 创建图像副本以进行绘制
        display_img = image.copy()
        
        if features:
            # 为不同特征类型设置不同的颜色
            colors = {
                'faces': (0, 255, 0),     # 绿色：正面人脸
                'profiles': (255, 0, 0),   # 蓝色：侧面人脸
                'bodies': (0, 0, 255),     # 红色：躯干
                'eyes': (255, 255, 0)      # 青色：眼睛
            }
            
            # 绘制检测到的特征
            for feature_type, color in colors.items():
                for (x, y, w, h) in features[feature_type]:
                    cv2.rectangle(display_img, (x, y), (x+w, y+h), color, 2)
        
        # 调整图像大小以适应窗口
        h, w = display_img.shape[:2]
        scale = min(window_size[0]/w, window_size[1]/h)
        new_size = (int(w*scale), int(h*scale))
        display_img = cv2.resize(display_img, new_size)
        
        # 创建固定大小的黑色背景
        background = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
        
        # 计算居中位置
        y_offset = (window_size[1] - new_size[1]) // 2
        x_offset = (window_size[0] - new_size[0]) // 2
        
        # 将调整后的图像放置在背景中心
        background[y_offset:y_offset+new_size[1], 
                  x_offset:x_offset+new_size[0]] = display_img
        
        # 显示图像
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, background)
        cv2.waitKey(1)  # 等待1毫秒
        
    except Exception as e:
        print(f"显示图像时出错: {str(e)}")

def detect_features(image_path, show_output=True, show_preview=True):
    """
    检测图片中的人脸和其他特征，支持多种特征类型的智能识别
    
    特征类型包括：
    - 正面人脸：使用多个级联分类器提高准确率
    - 侧面人脸：专门的侧脸检测器
    - 上半身躯干：适用于较远距离的人像
    - 眼睛：在检测到的人脸区域内进行精确定位
    
    Args:
        image_path (str): 输入图片路径，支持中文路径
        show_output (bool): 是否在控制台显示检测结果统计
        show_preview (bool): 是否显示实时预览窗口，包括特征标记
    
    Returns:
        dict: 包含所有检测到的特征信息的字典，格式如下：
            {
                'faces': [(x, y, w, h), ...],     # 正面人脸坐标
                'profiles': [(x, y, w, h), ...],  # 侧面人脸坐标
                'bodies': [(x, y, w, h), ...],    # 躯干坐标
                'eyes': [(x, y, w, h), ...]       # 眼睛坐标
            }
    """
    try:
        # 读取图片（支持中文路径）
        img_array = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法读取图片")
        
        if show_preview:
            show_processing_image(image, None, "原始图像")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = {
            'faces': [],
            'profiles': [],
            'bodies': [],
            'eyes': []
        }

        # 检测器配置（按精度和用途排序）
        cascades = {
            'faces': [
                # 正面人脸检测器
                ('haarcascade_frontalface_default.xml', 1.1, 5),
                ('haarcascade_frontalface_alt2.xml', 1.1, 3),
                ('haarcascade_frontalface_alt.xml', 1.2, 4),
            ],
            'profiles': [
                # 侧面人脸检测器
                ('haarcascade_profileface.xml', 1.1, 3),
            ],
            'bodies': [
                # 上半身躯干检测器
                ('haarcascade_upperbody.xml', 1.1, 3),
            ]
        }
        
        # 眼睛检测器
        eye_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
        )
        
        # 对每种特征进行检测
        for feature_type, cascade_configs in cascades.items():
            for cascade_file, scale_factor, min_neighbors in cascade_configs:
                cascade_path = os.path.join(cv2.data.haarcascades, cascade_file)
                if not os.path.exists(cascade_path):
                    continue
                    
                cascade = cv2.CascadeClassifier(cascade_path)
                if cascade.empty():
                    continue
                    
                # 根据特征类型调整检测参数
                min_size = (30, 30)  # 默认最小尺寸
                if feature_type == 'bodies':
                    min_size = (60, 60)  # 躯干需要更大的最小尺寸
                
                features = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size
                )
                
                if len(features) > 0:
                    result[feature_type].extend(features.tolist())
                    if feature_type != 'bodies':  # 对于躯干，我们继续检测其他可能的位置
                        break
        
        # 在检测到的正面人脸区域内检测眼睛
        for (x, y, w, h) in result['faces']:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            # 转换坐标到原图
            for (ex, ey, ew, eh) in eyes:
                result['eyes'].append((x+ex, y+ey, ew, eh))
        
        # 在检测到的侧面人脸区域内也检测眼睛
        for (x, y, w, h) in result['profiles']:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            # 转换坐标到原图
            for (ex, ey, ew, eh) in eyes:
                result['eyes'].append((x+ex, y+ey, ew, eh))

        # 在每次检测到新特征后更新显示
        if show_preview:
            show_processing_image(image, result, "特征检测结果")
        
        # 输出检测结果统计
        if show_output:
            detection_summary = []
            if result['faces']: detection_summary.append(f"{len(result['faces'])}个正面人脸")
            if result['profiles']: detection_summary.append(f"{len(result['profiles'])}个侧面人脸")
            if result['bodies']: detection_summary.append(f"{len(result['bodies'])}个躯干")
            if result['eyes']: detection_summary.append(f"{len(result['eyes'])}只眼睛")
            
            if detection_summary:
                print(f"✓ 检测到: {', '.join(detection_summary)}")
            else:
                print("⚠ 未检测到任何特征")
        
        return result
        
    except Exception as e:
        print(f'⚠ 错误: 特征检测失败 ({str(e)})')
        return {'faces': [], 'profiles': [], 'bodies': [], 'eyes': []}

def create_circular_image(input_path, output_path, size=(500, 500), dpi=300, fit_corners=False, input_format='PNG', output_format='PNG', face_center=False, detection_result=None):
    """
    创建圆形图片，支持多种处理模式和智能特征处理
    
    处理模式：
    1. 默认模式：根据长边等比例缩放并相切，适合一般图片
    2. 四角相切模式：确保原图完整显示且四角与圆形相切，适合方形图片
    3. 人脸检测模式：智能检测人脸位置并居中，适合头像处理
    
    特点：
    - 支持透明背景
    - 高质量缩放算法
    - 智能特征居中
    - 自动边界处理
    - 可自定义DPI
    
    Args:
        input_path (str): 输入图片路径，支持中文路径
        output_path (str): 输出图片路径
        size (tuple): 圆形画布大小（宽度，高度），单位为像素
        dpi (int): 输出图像DPI，影响最终分辨率和打印质量
        fit_corners (bool): 是否使用四角相切模式
        input_format (str): 输入图像格式，参见 INPUT_FORMATS
        output_format (str): 输出图像格式，参见 OUTPUT_FORMATS
        face_center (bool): 是否启用人脸检测和居中功能
        detection_result (dict): 已有的检测结果，可避免重复检测
    
    注意事项：
    1. 人脸检测模式下，未检测到特征时会使用图像中心
    2. 输出DPI会影响实际像素大小：实际像素 = size * (dpi/300)
    3. 建议使用PNG格式输出以保持透明背景
    """
    try:
        # 使用 PIL 打开图像（支持中文路径）
        img = Image.open(input_path)
        
        # 获取画布尺寸
        canvas_size = size[0]  # 使用宽度作为圆形直径
        
        # 根据DPI计算实际输出分辨率
        output_size = int(canvas_size * dpi / 300)  # 基准DPI为300
        
        if face_center:
            # 检测人脸和其他特征
            if detection_result is None:
                detection_result = detect_features(input_path, show_preview=False)
            
            faces = detection_result['faces']
            profiles = detection_result['profiles']
            total_faces = len(faces) + len(profiles)
            
            # 合并所有检测到的特征
            all_features = faces + profiles + detection_result['eyes']
            if not all_features and detection_result['bodies']:  # 如果没有检测到脸部特征但检测到躯干
                all_features = detection_result['bodies']
            
            if all_features:
                # 计算特征区域的边界框
                min_x = min(f[0] for f in all_features)
                min_y = min(f[1] for f in all_features)
                max_x = max(f[0]+f[2] for f in all_features)
                max_y = max(f[1]+f[3] for f in all_features)
                
                # 计算特征区域的中心点
                feature_center_x = (min_x + max_x) // 2
                feature_center_y = (min_y + max_y) // 2
                
                # 计算等比例缩放比例，确保图片填满圆形区域
                scale = output_size / min(img.width, img.height)
                
                # 计算缩放后的尺寸
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                
                # 缩放图片
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # 计算特征中心点在缩放后图片中的位置
                scaled_feature_x = int(feature_center_x * scale)
                scaled_feature_y = int(feature_center_y * scale)
                
                # 计算需要的偏移量，使特征中心点位于圆形中心
                offset_x = (output_size // 2) - scaled_feature_x
                offset_y = (output_size // 2) - scaled_feature_y
                
                # 创建透明背景画布
                canvas = Image.new('RGBA', (output_size, output_size), (0, 0, 0, 0))
                
                # 计算粘贴位置，确保特征居中
                paste_x = offset_x
                paste_y = offset_y
                
                # 调整粘贴位置，确保不会留下空白
                if paste_x > 0:
                    paste_x = 0
                elif paste_x + new_width < output_size:
                    paste_x = output_size - new_width
                    
                if paste_y > 0:
                    paste_y = 0
                elif paste_y + new_height < output_size:
                    paste_y = output_size - new_height
                
                # 粘贴图片
                canvas.paste(img, (paste_x, paste_y))
                print(f'✓ 已完成特征检测和居中处理')
            else:
                print(f'⚠ 未检测到特征: {os.path.basename(input_path)}，使用图像中心')
                # 未检测到特征时，使用默认的居中处理
                scale = output_size / min(img.width, img.height)
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # 创建透明背景画布
                canvas = Image.new('RGBA', (output_size, output_size), (0, 0, 0, 0))
                
                # 计算居中位置
                paste_x = (output_size - new_width) // 2
                paste_y = (output_size - new_height) // 2
                canvas.paste(img, (paste_x, paste_y))
            
        elif fit_corners:
            # 四角相切模式：保持图片完整性
            img_ratio = img.height / img.width
            
            if img_ratio > 1:  # 竖图
                new_height = output_size
                new_width = int(new_height / img_ratio)
            else:  # 横图或正方形
                new_width = output_size
                new_height = int(new_width * img_ratio)
            
            # 计算对角线长度并调整缩放比例
            diagonal = math.sqrt(new_width**2 + new_height**2)
            scale = output_size / diagonal
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)
            
            # 使用LANCZOS算法进行高质量缩放
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 创建透明背景画布
            canvas = Image.new('RGBA', (output_size, output_size), (0, 0, 0, 0))
            
            # 计算居中位置并粘贴图片
            paste_x = (output_size - new_width) // 2
            paste_y = (output_size - new_height) // 2
            canvas.paste(img, (paste_x, paste_y))
            
        else:
            # 默认模式：简单的长边对齐
            scale = output_size / max(img.width, img.height)
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            
            # 使用LANCZOS算法进行高质量缩放
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 创建透明背景画布
            canvas = Image.new('RGBA', (output_size, output_size), (0, 0, 0, 0))
            
            # 计算居中位置并粘贴图片
            paste_x = (output_size - new_width) // 2
            paste_y = (output_size - new_height) // 2
            canvas.paste(img, (paste_x, paste_y))
        
        # 创建和应用圆形遮罩
        mask = Image.new('L', (output_size, output_size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, output_size-1, output_size-1), fill=255)
        canvas.putalpha(mask)
        
        # 保存结果（带DPI信息）
        canvas.save(output_path, format=output_format, dpi=(dpi, dpi))
        
    except Exception as e:
        print(f'⚠ 错误: 处理图片时出错 ({str(e)})')
        raise

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

def process_directory(input_dir, output_dir, size=(500, 500), dpi=300, fit_corners=False, input_format='PNG', output_format='PNG', face_center=False, show_preview=True):
    """
    批量处理目录中的图片，支持多种处理模式和详细的结果统计
    
    功能特点：
    - 自动识别支持的图片格式
    - 保持原始文件组织结构
    - 智能错误处理和恢复
    - 详细的处理统计
    - 实时进度显示
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        size (tuple): 圆形画布大小（宽度，高度）
        dpi (int): 输出图像DPI，影响最终分辨率
        fit_corners (bool): 是否使用四角相切模式
        input_format (str): 输入图像格式
        output_format (str): 输出图像格式
        face_center (bool): 是否启用人脸检测和居中
        show_preview (bool): 是否显示处理过程预览
    
    处理流程：
    1. 扫描输入目录中的所有文件
    2. 过滤得到支持的图片文件
    3. 对每个图片执行处理
    4. 生成处理报告
    
    输出文件命名规则：
    - 默认模式：default_原文件名
    - 四角相切模式：fit_原文件名
    - 人脸检测模式：face_原文件名
    """
    try:
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
        all_files = []
        for file in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, file)):
                # 将文件名编码为UTF-8以支持中文
                all_files.append(file)
        
        # 从INPUT_FORMATS字典获取所有支持的文件扩展名
        supported_formats = []
        for extensions in INPUT_FORMATS.values():
            supported_formats.extend(extensions)
        
        # 对文件进行分类（支持的图片和不支持的文件）
        image_files = []
        unsupported_files = []
        for file in all_files:
            if any(file.lower().endswith(ext.lower()) for ext in supported_formats):
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
        processed_files = []    # 成功处理的文件
        failed_files = []       # 处理失败的文件
        no_feature_files = []   # 未检测到特征的文件
        
        # 处理每个图片
        for i, file in enumerate(image_files, 1):
            try:
                input_path = os.path.join(input_dir, file)
                
                # 如果是人脸检测模式，先检测特征
                detection_result = None
                if face_center:
                    detection_result = detect_features(input_path, show_preview=show_preview)
                    has_features = bool(detection_result['faces'] or detection_result['profiles'] or 
                                      detection_result['bodies'] or detection_result['eyes'])
                    if not has_features:
                        no_feature_files.append(file)
                
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
                    face_center=face_center,
                    detection_result=detection_result
                )
                
                # 显示处理进度
                print(f'[{i}/{len(image_files)}] ✓ {file} -> {output_filename}')
                processed_files.append(file)
                
                # 处理完成后关闭所有窗口
                if show_preview:
                    cv2.destroyAllWindows()
                
            except Exception as e:
                if show_preview:
                    cv2.destroyAllWindows()
                print(f'[{i}/{len(image_files)}] ✗ {file} - 处理失败: {str(e)}')
                failed_files.append((file, str(e)))
        
        # 显示最终处理结果统计
        print('\n处理结果统计:')
        print(f'输入文件夹总文件数: {len(all_files)} 个')
        print(f'可处理的图片: {len(image_files)} 个')
        print(f'成功处理: {len(processed_files)} 个')
        
        if face_center and no_feature_files:
            print(f'未检测到特征: {len(no_feature_files)} 个')
            print('未检测到特征的图片:')
            for file in no_feature_files:
                print(f'  - {file}')
        
        if failed_files:
            print(f'处理失败: {len(failed_files)} 个')
            print('失败的文件:')
            for file, error in failed_files:
                print(f'  - {file}: {error}')
        
        if unsupported_files:
            print(f'不支持的文件: {len(unsupported_files)} 个')
        
        print('\n处理完成!')
        
    except Exception as e:
        if show_preview:
            cv2.destroyAllWindows()
        print(f'⚠ 错误: 处理目录时出错 ({str(e)})')
        raise

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
以下是帮助指令和示例：

使用示例：
  # 使用默认设置（500px，PNG格式）
  python circle_image_converter.py
  
  # 指定输出尺寸为1000像素，使用四角相切模式
  python circle_image_converter.py --size 1000px --fit
  
  # 使用人脸检测模式，输出为300DPI的JPEG格式，并启用预览窗口
  python circle_image_converter.py --face --dpi 300 --output-format JPEG --preview
  
支持的输入格式：
  PNG, JPG/JPEG, BMP, GIF, WebP, TIFF, JFIF
  
支持的输出格式：
  PNG（推荐，支持透明背景）, JPEG, BMP, WebP, TIFF, JFIF
  
尺寸单位支持：
  px（像素）, mm（毫米）, cm（厘米）
  示例: 500px, 50mm, 5cm

命令行选项：
--size：设置圆形画布尺寸，支持以下格式：
  - 像素：500px（默认）
  - 毫米：50mm
  - 厘米：5cm
--dpi：设置输出图片DPI，默认300
--fit：启用四角相切模式，确保原图完整显示且四角与圆形相切
--face：启用人脸检测模式，自动检测人脸并居中
--input-format：指定输入图片格式，默认PNG
--output-format：指定输出图片格式，默认PNG
--preview：启用处理过程预览窗口
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
    parser.add_argument('--preview', action='store_true',
                      help='启用处理过程预览窗口')
    
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
            face_center=args.face,
            show_preview=args.preview
        )
        
        # 处理完成后打开输出文件夹
        open_folder(output_dir)
        
    except ValueError as e:
        print(f'错误: {str(e)}')
        sys.exit(1)
    except Exception as e:
        cv2.destroyAllWindows()  # 确保在发生错误时关闭所有窗口
        print(f'发生错误: {str(e)}')
        sys.exit(1)

# 程序入口点
if __name__ == '__main__':
    main()
