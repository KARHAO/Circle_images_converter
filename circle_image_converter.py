# Import required libraries | 导入所需的库 2025-3-11
from PIL import Image, ImageDraw, ImageFont  # PIL library for image processing | PIL库用于图像处理
import os                         # File and directory operations | 文件和目录操作
import argparse                   # Command line argument parsing | 命令行参数解析
import re                         # Regular expression matching | 正则表达式匹配
import sys                        # System-specific operations | 系统相关操作
import math                       # Mathematical calculations | 数学计算
import cv2                        # OpenCV for face and feature detection | OpenCV用于人脸和特征检测
import numpy as np               # Array operations | 数组操作
import tkinter as tk            # GUI toolkit | GUI工具包
from tkinter import filedialog  # File dialog interface | 文件对话框界面
import subprocess              # System command operations | 系统命令操作

# Configure console output encoding for Windows | 设置Windows系统的控制台输出编码
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Initialize face detector | 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Supported input formats and their extensions | 支持的输入格式及其文件扩展名
INPUT_FORMATS = {
    'JPG': ['.jpg', '.jpeg'],    # JPEG format: lossy compression, suitable for photos | JPEG格式：有损压缩，适合照片
    'PNG': ['.png'],             # PNG format: lossless with transparency | PNG格式：无损格式，支持透明背景
    'BMP': ['.bmp'],             # BMP format: uncompressed bitmap | BMP格式：无压缩位图
    'WEBP': ['.webp'],           # WebP format: modern format by Google | WebP格式：Google开发的现代格式
    'GIF': ['.gif'],             # GIF format: supports animation (first frame only) | GIF格式：支持动画（仅处理第一帧）
    'TIFF': ['.tiff', '.tif'],   # TIFF format: professional format | TIFF格式：专业图像格式
    'JFIF': ['.jfif'],           # JFIF format: JPEG variant | JFIF格式：JPEG文件交换格式
}

# Supported output formats and their extensions | 支持的输出格式及其文件扩展名
OUTPUT_FORMATS = {
    'PNG': '.png',    # PNG format (recommended: supports transparency) | PNG格式（推荐：支持透明背景）
    'JPEG': '.jpg',   # JPEG format (lossy compression, small file size) | JPEG格式（有损压缩，文件小）
    'BMP': '.bmp',    # BMP format (uncompressed, large file size) | BMP格式（无压缩，文件大）
    'TIFF': '.tiff',  # TIFF format (professional use) | TIFF格式（专业用途）
    'WEBP': '.webp',  # WebP format (modern, good compression) | WebP格式（现代格式，压缩好）
    'JFIF': '.jfif',  # JFIF format (JPEG variant) | JFIF格式（JPEG变体）
}

def get_supported_input_extensions():
    """
    获取所有支持的输入文件扩展名
    
    Returns:
        list: 支持的文件扩展名列表（小写）
        
    用途：
        用于文件过滤，确保只处理支持的图片格式
    """
    extensions = []
    for format_exts in INPUT_FORMATS.values():
        extensions.extend(format_exts)
    return [ext.lower() for ext in extensions]

def get_output_extension(format_name):
    """
    获取指定输出格式的文件扩展名
    
    Args:
        format_name (str): 输出格式名称，如'PNG', 'JPEG'等
    
    Returns:
        str: 对应的文件扩展名（如'.png', '.jpg'）
        
    Raises:
        ValueError: 如果指定的格式不支持
        
    用途：
        确保输出文件使用正确的扩展名
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
        
    用途：
        提供灵活的尺寸输入方式，支持多种单位
        自动转换为像素值，方便后续处理
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
        dpi (int): DPI值，影响物理尺寸计算
    
    Returns:
        float: 转换后的尺寸值
        
    用途：
        用于显示和输出时的单位转换
        帮助用户理解实际物理尺寸
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
        circle_diameter (int): 圆的直径（像素）
    
    Returns:
        int: 内接正方形的边长（像素）
        
    用途：
        用于计算图片缩放尺寸
        确保图片在圆形区域内完全显示
        适用于需要保持图片完整性的场景
    """
    # 内接正方形的边长等于圆直径除以根号2
    return int(circle_diameter / math.sqrt(2))

def calculate_circumscribed_square_size(circle_diameter):
    """
    计算外接正方形的边长（正方形在圆外）
    
    Args:
        circle_diameter (int): 圆的直径（像素）
    
    Returns:
        int: 外接正方形的边长（像素）
        
    用途：
        用于计算图片缩放尺寸
        确保圆形区域被图片完全覆盖
        适用于需要填满圆形区域的场景
    """
    # 外接正方形的边长等于圆的直径
    return circle_diameter

def show_processing_image(image, features=None, window_name="处理进度预览", window_size=(800, 600), wait_key=1):
    """
    显示处理过程中的图像预览，包括特征标记和实时更新
    
    显示特点：
    - 固定大小窗口：保持界面统一性
    - 自动缩放：适应不同尺寸的图片
    - 特征标记：使用不同颜色区分不同特征
    - 居中显示：提供良好的视觉体验
    - 支持中英文：国际化支持
    - 实时更新：直观展示处理过程
    
    特征标记颜色说明：
    - 绿色：正面人脸（最常见的特征）
    - 蓝色：侧面人脸（配合正面检测）
    - 红色：躯干（在人脸检测失败时使用）
    - 青色：眼睛（精确定位面部特征）
    - 紫色：嘴巴（补充面部特征检测）
    
    Args:
        image (numpy.ndarray): 要显示的图像
        features (dict): 检测到的特征信息，包含坐标
        window_name (str): 窗口标题
        window_size (tuple): 窗口大小 (宽, 高)
        wait_key (int): 等待键盘输入的时间（毫秒）
            0: 无限等待直到按键
            >0: 等待指定毫秒数
    
    Returns:
        int: 按键值（-1表示出错）
        
    用途：
        实时预览处理效果
        在手动模式下等待用户输入
        显示特征检测结果
        提供直观的操作指引
    """
    try:
        # 创建图像副本以进行绘制
        display_img = image.copy()
        
        # 为不同特征类型设置不同的颜色和说明文字
        feature_info = {
            'faces': {
                'color': (0, 255, 0),     # 绿色
                'label': 'Front Face'
            },
            'profiles': {
                'color': (255, 0, 0),     # 蓝色
                'label': 'Profile Face'
            },
            'bodies': {
                'color': (0, 0, 255),     # 红色
                'label': 'Upper Body'
            },
            'eyes': {
                'color': (255, 255, 0),   # 青色
                'label': 'Eyes'
            },
            'mouths': {
                'color': (255, 0, 255),   # 紫色
                'label': 'Mouth'
            }
        }
        
        if features:
            # 绘制检测到的特征
            for feature_type, info in feature_info.items():
                if feature_type in features:
                    for (x, y, w, h) in features[feature_type]:
                        # 绘制矩形框
                        cv2.rectangle(display_img, (x, y), (x+w, y+h), info['color'], 2)
                        
                        # 在矩形框上方添加标签文字
                        label_size = cv2.getTextSize(info['label'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        
                        # 确保标签位置在图像范围内
                        label_x = max(x, 0)
                        label_y = max(y - 10, label_size[1])
                        
                        # 绘制半透明背景
                        overlay = display_img.copy()
                        cv2.rectangle(overlay, 
                                    (label_x, label_y - label_size[1] - 5),
                                    (label_x + label_size[0], label_y + 5),
                                    info['color'], -1)
                        cv2.addWeighted(overlay, 0.7, display_img, 0.3, 0, display_img)
                        
                        # 绘制文字
                        cv2.putText(display_img, info['label'],
                                  (label_x, label_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (255, 255, 255), 1, cv2.LINE_AA)
        
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
        
        # 如果是手动模式，添加操作说明
        if wait_key == 0:
            # 将OpenCV图像转换为PIL图像以支持中文
            background_pil = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(background_pil)
            
            # 尝试加载系统中文字体
            try:
                if sys.platform.startswith('win'):
                    font = ImageFont.truetype("simhei.ttf", 24)  # Windows下的黑体
                else:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 24)  # Linux下的字体
            except:
                # 如果找不到系统字体，使用默认字体
                font = ImageFont.load_default()
            
            # 添加检测框说明
            legend_texts = [
                "检测框说明 / Detection Legend:",
                "绿色框 / Green: 正面人脸 / Front Face",
                "蓝色框 / Blue: 侧面人脸 / Profile Face",
                "红色框 / Red: 躯干 / Upper Body",
                "青色框 / Cyan: 眼睛 / Eyes",
                "紫色框 / Purple: 嘴巴 / Mouth"
            ]
            
            # 在顶部显示检测框说明
            y = 10
            for text in legend_texts:
                # 计算文本大小
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # 绘制半透明背景
                rect_coords = [10, y, text_width + 20, y + text_height + 5]
                rect_img = Image.new('RGBA', background_pil.size, (0, 0, 0, 0))
                rect_draw = ImageDraw.Draw(rect_img)
                rect_draw.rectangle(rect_coords, fill=(0, 0, 0, 128))
                background_pil = Image.alpha_composite(background_pil.convert('RGBA'), rect_img)
                draw = ImageDraw.Draw(background_pil)
                
                # 绘制文本
                draw.text((10, y), text, font=font, fill=(255, 255, 255))
                y += 35
            
            # 添加操作说明
            instructions = [
                "按键选择处理模式 / Press key to select mode:",
                "1: 人脸检测模式 / Face Detection Mode",
                "2: 默认模式 / Default Mode",
                "3: 四角相切模式 / Fit Corners Mode",
                "ESC: 跳过当前图片 / Skip Current Image"
            ]
            
            # 在底部显示操作说明
            y = window_size[1] - 35 * len(instructions) - 10
            for text in instructions:
                # 计算文本大小
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # 绘制半透明背景
                rect_coords = [10, y, text_width + 20, y + text_height + 5]
                rect_img = Image.new('RGBA', background_pil.size, (0, 0, 0, 0))
                rect_draw = ImageDraw.Draw(rect_img)
                rect_draw.rectangle(rect_coords, fill=(0, 0, 0, 128))
                background_pil = Image.alpha_composite(background_pil.convert('RGBA'), rect_img)
                draw = ImageDraw.Draw(background_pil)
                
                # 绘制文本
                draw.text((10, y), text, font=font, fill=(255, 255, 255))
                y += 35
            
            # 将PIL图像转回OpenCV格式
            background = cv2.cvtColor(np.array(background_pil), cv2.COLOR_RGB2BGR)
        
        # 显示图像
        cv2.imshow(window_name, background)
        key = cv2.waitKey(wait_key) & 0xFF
        
        return key
        
    except Exception as e:
        print(f"显示图像时出错: {str(e)}")
        return -1

def detect_features(image_path, show_output=True, show_preview=True, preview_size=(800, 600)):
    """
    检测图片中的人脸和其他特征，支持多种特征类型的智能识别
    
    特征类型包括：
    - 正面人脸：使用多个级联分类器提高准确率
    - 侧面人脸：专门的侧脸检测器，提高检测全面性
    - 上半身躯干：适用于较远距离的人像，作为备选特征
    - 眼睛：在检测到的人脸区域内进行精确定位，提高准确性
    - 嘴巴：在检测到的人脸区域内进行精确定位，增强特征识别
    
    检测策略：
    1. 优先使用多个人脸检测器，提高准确率
    2. 在人脸区域内进行眼睛和嘴巴检测
    3. 对正面和侧面人脸分别进行特征检测
    4. 使用躯干检测作为备选方案
    
    Args:
        image_path (str): 输入图片路径，支持中文路径
        show_output (bool): 是否在控制台显示检测结果统计
        show_preview (bool): 是否显示实时预览窗口
        preview_size (tuple): 预览窗口大小 (宽, 高)
    
    Returns:
        dict: 包含所有检测到的特征信息的字典，格式如下：
            {
                'faces': [(x, y, w, h), ...],     # 正面人脸坐标
                'profiles': [(x, y, w, h), ...],  # 侧面人脸坐标
                'bodies': [(x, y, w, h), ...],    # 躯干坐标
                'eyes': [(x, y, w, h), ...],      # 眼睛坐标
                'mouths': [(x, y, w, h), ...]     # 嘴巴坐标
            }
    
    注意事项：
    1. 检测结果可能受图片质量、光线、角度等因素影响
    2. 某些特征可能无法检测到，此时返回空列表
    3. 坐标值均为像素单位，相对于原图
    4. 检测参数已经过优化，平衡了准确率和速度
    """
    try:
        # 读取图片（支持中文路径）
        img_array = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法读取图片")
        
        if show_preview:
            show_processing_image(image, None, "原始图像", preview_size)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = {
            'faces': [],
            'profiles': [],
            'bodies': [],
            'eyes': [],
            'mouths': []  # 添加嘴巴检测结果
        }

        # 检测器配置（按精度和用途排序）
        cascades = {
            'faces': [
                ('haarcascade_frontalface_default.xml', 1.1, 5),
                ('haarcascade_frontalface_alt2.xml', 1.1, 3),
                ('haarcascade_frontalface_alt.xml', 1.2, 4),
            ],
            'profiles': [
                ('haarcascade_profileface.xml', 1.1, 3),
            ],
            'bodies': [
                ('haarcascade_upperbody.xml', 1.1, 3),
            ]
        }
        
        # 眼睛检测器
        eye_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
        )
        
        # 嘴巴检测器
        mouth_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, 'haarcascade_smile.xml')
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
        
        # 在检测到的正面人脸区域内检测眼睛和嘴巴
        for (x, y, w, h) in result['faces']:
            # 提取人脸区域
            roi_gray = gray[y:y+h, x:x+w]
            
            # 检测眼睛
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            # 转换眼睛坐标到原图
            for (ex, ey, ew, eh) in eyes:
                result['eyes'].append((x+ex, y+ey, ew, eh))
            
            # 检测嘴巴（主要在人脸下半部分）
            roi_mouth = roi_gray[h//2:, :]  # 只在人脸下半部分检测
            mouths = mouth_cascade.detectMultiScale(
                roi_mouth,
                scaleFactor=1.1,
                minNeighbors=20,  # 增加此值以减少误检
                minSize=(25, 15)  # 设置最小尺寸
            )
            # 转换嘴巴坐标到原图
            for (mx, my, mw, mh) in mouths:
                result['mouths'].append((x+mx, y+h//2+my, mw, mh))
        
        # 在检测到的侧面人脸区域内也检测眼睛和嘴巴
        for (x, y, w, h) in result['profiles']:
            # 提取人脸区域
            roi_gray = gray[y:y+h, x:x+w]
            
            # 检测眼睛
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            # 转换眼睛坐标到原图
            for (ex, ey, ew, eh) in eyes:
                result['eyes'].append((x+ex, y+ey, ew, eh))
            
            # 检测嘴巴
            roi_mouth = roi_gray[h//2:, :]  # 只在人脸下半部分检测
            mouths = mouth_cascade.detectMultiScale(
                roi_mouth,
                scaleFactor=1.1,
                minNeighbors=20,
                minSize=(25, 15)
            )
            # 转换嘴巴坐标到原图
            for (mx, my, mw, mh) in mouths:
                result['mouths'].append((x+mx, y+h//2+my, mw, mh))

        # 在每次检测到新特征后更新显示
        if show_preview:
            show_processing_image(image, result, "特征检测结果", preview_size)
        
        # 输出检测结果统计
        if show_output:
            detection_summary = []
            if result['faces']: detection_summary.append(f"{len(result['faces'])}个正面人脸")
            if result['profiles']: detection_summary.append(f"{len(result['profiles'])}个侧面人脸")
            if result['bodies']: detection_summary.append(f"{len(result['bodies'])}个躯干")
            if result['eyes']: detection_summary.append(f"{len(result['eyes'])}只眼睛")
            if result['mouths']: detection_summary.append(f"{len(result['mouths'])}个嘴巴")
            
            if detection_summary:
                print(f"✓ 已成功检测到特征")
            else:
                print("⚠ 未检测到任何特征")
        
        return result
        
    except Exception as e:
        print(f'⚠ 错误: 特征检测失败 ({str(e)})')
        return {'faces': [], 'profiles': [], 'bodies': [], 'eyes': [], 'mouths': []}

def create_circular_image(input_path, output_path, size=(500, 500), dpi=300, fit_corners=False, input_format='PNG', output_format='PNG', face_center=False, detection_result=None):
    """
    创建圆形图片，支持多种处理模式和智能特征处理
    
    处理模式说明：
    1. 默认模式：
       - 根据长边等比例缩放并相切
       - 适合一般图片
       - 保持图片比例，可能裁剪部分内容
    
    2. 四角相切模式：
       - 确保原图完整显示且四角与圆形相切
       - 适合方形图片和徽标
       - 保持图片完整性，可能留有空白
    
    3. 人脸检测模式：
       - 智能检测人脸位置并居中
       - 适合头像处理
       - 自动调整缩放和位置
    
    特点和优势：
    - 支持透明背景：适合各种使用场景
    - 高质量缩放：使用LANCZOS算法
    - 智能特征居中：提高头像质量
    - 自动边界处理：避免图片变形
    - 可自定义DPI：满足不同输出需求
    
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
    
    技术细节：
    1. 图像处理流程：
       - 读取原图 -> 特征检测 -> 缩放计算 -> 位置调整 -> 圆形裁剪 -> 保存
    
    2. 特征检测优化：
       - 优先使用人脸特征
       - 使用眼睛和嘴巴位置微调
       - 躯干检测作为备选
    
    3. 缩放策略：
       - 默认模式：保持比例，适应圆形
       - 四角相切：完整显示，可能留白
       - 人脸模式：根据特征位置动态调整
    
    4. 输出优化：
       - 自动调整DPI
       - 保持透明通道
       - 高质量保存
    
    注意事项：
    1. 人脸检测模式下，未检测到特征时会使用图像中心
    2. 输出DPI会影响实际像素大小：实际像素 = size * (dpi/300)
    3. 建议使用PNG格式输出以保持透明背景
    4. 处理大图片时可能需要较多内存
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
            all_features = faces + profiles + detection_result['eyes'] + detection_result['mouths']
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
                print(f'⚠ 未检测到特征: {os.path.basename(input_path)}，使用默认模式')
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

def process_directory(input_dir, output_dir, size=(500, 500), dpi=300, fit_corners=False, 
                     input_format='PNG', output_format='PNG', face_center=False, 
                     show_preview=True, add_prefix=True, manual_mode=False):
    """
    批量处理目录中的图片，支持多种处理模式和详细的结果统计
    
    功能特点：
    1. 文件处理：
       - 自动识别支持的图片格式
       - 保持原始文件组织结构
       - 智能文件名处理
       - 支持中文路径
    
    2. 错误处理：
       - 智能错误检测和恢复
       - 详细的错误信息记录
       - 继续处理其他文件
       - 最终统计报告
    
    3. 进度显示：
       - 实时处理进度
       - 预览窗口（可选）
       - 特征检测结果
       - 处理统计信息
    
    4. 手动模式：
       - 交互式处理选择
       - 实时预览效果
       - 灵活的处理方式
       - 可跳过特定图片
    
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
        add_prefix (bool): 是否在输出文件名前添加模式前缀
        manual_mode (bool): 是否启用手动选择模式
    
    处理流程：
    1. 初始化：
       - 创建输出目录
       - 设置处理模式
       - 准备文件列表
    
    2. 文件分类：
       - 识别支持的格式
       - 过滤不支持的文件
       - 统计文件数量
    
    3. 批量处理：
       - 逐个处理图片
       - 实时显示进度
       - 错误处理和恢复
       - 结果统计
    
    4. 结果报告：
       - 处理成功数量
       - 失败文件列表
       - 未检测特征统计
       - 不支持格式统计
    
    注意事项：
    1. 大量文件处理时建议关闭预览窗口以提高速度
    2. 手动模式下需要用户交互，处理时间会较长
    3. 建议定期检查输出目录的空间占用
    4. 处理失败的文件会被记录但不会中断整体处理
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 根据处理模式设置输出文件名前缀
        mode_prefix = ""
        if add_prefix:
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
                
                # 当前图片的处理模式
                current_face_center = face_center
                current_fit_corners = fit_corners
                
                # 如果是人脸检测模式，先检测特征
                detection_result = None
                if face_center:
                    preview_size = (1024, 768) if manual_mode else (800, 600)
                    
                    # 在手动模式下创建固定窗口
                    if manual_mode and i == 1:
                        cv2.namedWindow("手动模式 - 选择处理方式", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("手动模式 - 选择处理方式", preview_size[0], preview_size[1])
                    
                    detection_result = detect_features(
                        input_path, 
                        show_preview=show_preview and not manual_mode,
                        preview_size=preview_size
                    )
                    
                    # 在手动模式下显示预览并等待用户选择
                    if manual_mode:
                        img = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                        key = show_processing_image(
                            img,
                            detection_result,
                            window_name="手动模式 - 选择处理方式",
                            window_size=preview_size,
                            wait_key=0
                        )
                        
                        # 根据用户选择设置处理模式
                        if key == ord('1'):
                            current_face_center = True
                            current_fit_corners = False
                            print(f"已选择人脸检测模式")
                        elif key == ord('2'):
                            current_face_center = False
                            current_fit_corners = False
                            print(f"已选择默认模式")
                        elif key == ord('3'):
                            current_face_center = False
                            current_fit_corners = True
                            print(f"已选择四角相切模式")
                        elif key == 27:  # ESC键
                            print(f"已跳过处理：{file}")
                            continue
                        
                        cv2.destroyWindow("手动模式 - 选择处理方式")
                    
                    has_features = bool(detection_result['faces'] or detection_result['profiles'] or 
                                      detection_result['bodies'] or detection_result['eyes'] or detection_result['mouths'])
                    if not has_features:
                        no_feature_files.append(file)
                
                # 添加模式前缀到输出文件名
                filename = os.path.splitext(file)[0]
                mode_prefix = ""
                if add_prefix:
                    if current_face_center:
                        mode_prefix = "face_"
                    elif current_fit_corners:
                        mode_prefix = "fit_"
                    else:
                        mode_prefix = "default_"
                
                output_filename = f"{mode_prefix}{filename}.{output_format.lower()}"
                output_path = os.path.join(output_dir, output_filename)
                
                # 创建圆形图片
                create_circular_image(
                    input_path=input_path,
                    output_path=output_path,
                    size=size,
                    dpi=dpi,
                    fit_corners=current_fit_corners,
                    input_format=input_format,
                    output_format=output_format,
                    face_center=current_face_center,
                    detection_result=detection_result
                )
                
                # 显示处理进度
                print(f'[{i}/{len(image_files)}] ✓ {file} -> {output_filename}')
                processed_files.append(file)
                
                # 处理完成后关闭所有窗口（仅在非手动模式下）
                if show_preview and not manual_mode:
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

def show_welcome_message():
    """
    显示欢迎信息和使用指南
    """
    print("\n" + "="*50)
    print("欢迎使用圆形图片转换工具")
    print("="*50)
    print("\n该工具可以将各种格式的图片转换为带透明背景的圆形图片")
    print("支持三种处理模式和多种图片格式")
    
    print("\n主要特点:")
    print("  1. 三种处理模式：默认模式、四角相切模式和人脸检测模式")
    print("  2. 支持多种图片格式：PNG、JPG/JPEG、BMP、GIF、WebP等")
    print("  3. 智能人脸检测：自动识别并居中人脸")
    print("  4. 灵活的尺寸设置：支持像素(px)、毫米(mm)、厘米(cm)")
    print("  5. 批量处理：支持整个文件夹的图片批量转换")
    
    print("\n可用命令:")
    print("  1: 默认模式 - 保持图片比例，长边与圆形相切")
    print("  2: 四角相切模式 - 确保原图完整显示")
    print("  3: 人脸检测模式 - 自动检测人脸并居中")
    print("  4: 人脸检测手动模式 - 可以手动选择每张图片的处理方式")
    print("  5: 高级选项 - 自定义尺寸、DPI等参数")
    print("  h: 显示帮助信息")
    print("  q: 退出程序")
    
    print("\n" + "="*50)

def show_help():
    """
    显示详细帮助信息
    """
    print("\n详细帮助信息:")
    print("\n1. 默认模式")
    print("   描述: 保持图片比例，并使图片的长边与圆形相切。")
    print("   适用: 一般图片处理")
    
    print("\n2. 四角相切模式")
    print("   描述: 确保原图完整显示，并使图片的四个角与圆形相切。")
    print("   适用: 方形图片和徽标")
    
    print("\n3. 人脸检测模式")
    print("   描述: 自动识别人脸位置并居中。")
    print("   适用: 头像处理")
    
    print("\n4. 人脸检测手动模式")
    print("   描述: 可以手动选择每张图片的处理方式。")
    print("   使用方法: 按1选择人脸模式，按2选择默认模式，按3选择四角相切模式，按ESC跳过")
    
    print("\n5. 支持的文件格式:")
    print("   输入: PNG, JPG/JPEG, BMP, GIF, WebP, TIFF, JFIF")
    print("   输出: PNG(推荐), JPEG, BMP, WebP, TIFF, JFIF")
    
    print("\n6. 高级选项说明:")
    print("   尺寸设置: 支持像素(px)、毫米(mm)、厘米(cm)，如500px, 50mm, 5cm")
    print("   DPI设置: 影响最终输出分辨率，默认为300")
    print("   预览功能: 可以实时查看处理过程")
    print("   前缀功能: 在输出文件名前添加模式前缀")
    
    print("\n" + "="*50)

def get_user_choice():
    """
    获取用户输入的命令
    """
    print("\n请输入命令数字或字母 (h显示帮助, q退出):")
    return input("> ").strip().lower()

def run_with_args(args_list):
    """
    使用指定参数运行程序
    
    Args:
        args_list: 命令行参数列表
    """
    # 保存原始参数
    original_argv = sys.argv.copy()
    
    try:
        # 替换命令行参数
        sys.argv = ['circle_image_converter.py'] + args_list
        
        # 创建新的参数解析器
        parser = argparse.ArgumentParser()
        parser.add_argument('--size', default='500px')
        parser.add_argument('--dpi', type=int, default=300)
        parser.add_argument('--fit', action='store_true')
        parser.add_argument('--input-format', default='PNG', choices=list(INPUT_FORMATS.keys()))
        parser.add_argument('--output-format', default='PNG', choices=list(OUTPUT_FORMATS.keys()))
        parser.add_argument('--face', action='store_true')
        parser.add_argument('--preview', action='store_true')
        parser.add_argument('--prefix', action='store_true')
        parser.add_argument('--manual', action='store_true')
        
        # 解析参数
        args = parser.parse_args()
        
        # 选择输入文件夹
        input_dir = select_folder('选择输入文件夹')
        if not input_dir:
            print('未选择输入文件夹，操作取消')
            return
        
        # 选择输出文件夹
        output_dir = select_folder('选择输出文件夹')
        if not output_dir:
            print('未选择输出文件夹，操作取消')
            return
        
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
            show_preview=args.preview,
            add_prefix=args.prefix,
            manual_mode=args.manual and args.face
        )
        
        # 处理完成后打开输出文件夹
        open_folder(output_dir)
        
    except ValueError as e:
        print(f'错误: {str(e)}')
    except Exception as e:
        cv2.destroyAllWindows()
        print(f'发生错误: {str(e)}')
    finally:
        # 恢复原始参数
        sys.argv = original_argv.copy()

def get_advanced_options():
    """
    获取用户的高级选项设置
    
    Returns:
        list: 参数列表
    """
    args = []
    
    # 获取尺寸
    print("\n请输入圆形画布尺寸 (如: 500px、50mm、5cm，直接回车使用默认值500px):")
    size = input("> ").strip()
    if size:
        args.extend(['--size', size])
    
    # 获取DPI
    print("\n请输入输出DPI (直接回车使用默认值300):")
    dpi = input("> ").strip()
    if dpi and dpi.isdigit():
        args.extend(['--dpi', dpi])
    
    # 获取输入格式
    print(f"\n请选择输入格式 (可选: {', '.join(INPUT_FORMATS.keys())}, 直接回车使用默认值PNG):")
    input_format = input("> ").strip().upper()
    if input_format in INPUT_FORMATS:
        args.extend(['--input-format', input_format])
    
    # 获取输出格式
    print(f"\n请选择输出格式 (可选: {', '.join(OUTPUT_FORMATS.keys())}, 直接回车使用默认值PNG):")
    output_format = input("> ").strip().upper()
    if output_format in OUTPUT_FORMATS:
        args.extend(['--output-format', output_format])
    
    # 询问是否启用预览
    print("\n是否启用处理过程预览? (y/n, 直接回车为n):")
    preview = input("> ").strip().lower()
    if preview == 'y':
        args.append('--preview')
    
    # 询问是否添加前缀
    print("\n是否在输出文件名前添加模式前缀? (y/n, 直接回车为n):")
    prefix = input("> ").strip().lower()
    if prefix == 'y':
        args.append('--prefix')
    
    return args

def main():
    """
    主函数：显示介绍信息，等待用户输入指令后执行相应操作
    """
    try:
        # 显示欢迎信息
        show_welcome_message()
        
        while True:
            # 获取用户选择
            choice = get_user_choice()
            
            if choice == 'q':
                print("退出程序")
                break
                
            elif choice == 'h':
                show_help()
                
            elif choice == '1':
                # 默认模式
                print("\n选择了默认模式")
                run_with_args([])
                
            elif choice == '2':
                # 四角相切模式
                print("\n选择了四角相切模式")
                run_with_args(['--fit'])
                
            elif choice == '3':
                # 人脸检测模式
                print("\n选择了人脸检测模式")
                run_with_args(['--face'])
                
            elif choice == '4':
                # 人脸检测手动模式
                print("\n选择了人脸检测手动模式")
                run_with_args(['--face', '--manual'])
                
            elif choice == '5':
                # 高级选项
                print("\n选择了高级选项")
                advanced_args = get_advanced_options()
                
                # 询问额外的模式
                print("\n请选择处理模式:")
                print("1: 默认模式")
                print("2: 四角相切模式")
                print("3: 人脸检测模式")
                print("4: 人脸检测手动模式")
                mode = input("> ").strip()
                
                if mode == '2':
                    advanced_args.append('--fit')
                elif mode == '3':
                    advanced_args.append('--face')
                elif mode == '4':
                    advanced_args.extend(['--face', '--manual'])
                
                run_with_args(advanced_args)
                
            else:
                print("无效的命令，请重新输入")
                
    except KeyboardInterrupt:
        print("\n程序被用户终止")
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        
# 程序入口点
if __name__ == '__main__':
    main()
