# 圆形图片转换工具 Circle_image_converter

一个功能强大的批量圆形图片转换工具，支持多种处理模式和智能特征检测。

## 主要特点

- 🎯 三种处理模式：默认模式、四角相切模式和人脸检测模式
- 🖼️ 支持多种图片格式：PNG、JPG/JPEG、BMP、GIF、WebP、TIFF、JFIF
- 👤 智能人脸检测：自动识别并居中人脸
- 🔍 特征检测：支持正面人脸、侧面人脸、眼睛和嘴巴检测
- 📏 灵活的尺寸设置：支持像素(px)、毫米(mm)、厘米(cm)
- 🎨 透明背景支持：完美适配各种使用场景
- 🔄 批量处理：支持整个文件夹的图片批量转换
- 👁️ 实时预览：可以预览处理效果
- ✨ 手动模式：支持交互式选择处理方式

## 处理模式示例

### 1. 默认模式--default
默认模式会保持图片比例，并使图片的长边与圆形相切。

![默认模式示例](https://github.com/KARHAO/Circle_images_converter/blob/main/examples/default_demo.png)

### 2. 四角相切模式--fit
四角相切模式会确保原图完整显示，并使图片的四个角与圆形相切。

![四角相切模式示例](https://github.com/KARHAO/Circle_images_converter/blob/main/examples/fit_demo.png)

### 3. 人脸检测模式--face
人脸检测模式会自动识别人脸位置并居中，适合处理头像。

![人脸检测模式示例](https://github.com/KARHAO/Circle_images_converter/blob/main/examples/face_demo.png)

## 使用方法

### 1. 安装PYTHON

### 2. 安装依赖
```bash
# CMD运行
pip install -r /path/to/requirements.txt
```
### 2. 运行程序
```bash
# CMD运行
python circle_image_converter.py
```

### 基本命令

```bash
# 默认模式
python circle_image_converter.py

# 四角相切模式
python circle_image_converter.py --fit

# 人脸检测模式
python circle_image_converter.py --face

# 人脸检测模式（手动选择）
python circle_image_converter.py --face --manual
```

### 高级选项

```bash
# 自定义尺寸和DPI
python circle_image_converter.py --size 1000px --dpi 300

# 添加模式前缀到输出文件名
python circle_image_converter.py --prefix

# 启用预览窗口
python circle_image_converter.py --preview
```

## 命令行参数说明

- `--size`: 设置圆形画布尺寸（默认：500px）
  - 支持单位：px（像素）、mm（毫米）、cm（厘米）
  - 示例：`--size 500px`、`--size 50mm`、`--size 5cm`

- `--dpi`: 设置输出图片DPI（默认：300）

- `--fit`: 启用四角相切模式

- `--face`: 启用人脸检测模式

- `--preview`: 启用处理过程预览窗口

- `--prefix`: 在输出文件名添加模式前缀

- `--manual`: 启用手动选择模式（需要同时使用`--face`）

- `--input-format`: 指定输入图片格式（默认：PNG）

- `--output-format`: 指定输出图片格式（默认：PNG）

## 手动模式操作说明

在手动模式下（使用`--face --manual`参数），您可以：

1. 按 `1` 键：选择人脸检测模式
2. 按 `2` 键：选择默认模式
3. 按 `3` 键：选择四角相切模式
4. 按 `ESC` 键：跳过当前图片

## 支持的文件格式

### 输入格式
- PNG
- JPG/JPEG
- BMP
- GIF（仅处理第一帧）
- WebP
- TIFF
- JFIF

### 输出格式
- PNG（推荐，支持透明背景）
- JPEG
- BMP
- WebP
- TIFF
- JFIF

## 注意事项

1. 建议使用PNG格式输出以保持透明背景

2. 人脸检测效果可能受图片质量、光线、角度等因素影响

3. 处理大量图片时建议关闭预览窗口以提高处理速度

4. 输出DPI会影响实际像素大小：实际像素 = size * (dpi/300)