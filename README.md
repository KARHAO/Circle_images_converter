# 圆形图片转换工具

一个功能强大的图片转换工具，可以将各种格式的图片转换为带透明背景的圆形图片。

## 功能特点

- 支持多种输入格式：PNG, JPG, JPEG, BMP, GIF, WebP, TIFF, JFIF
- 支持多种输出格式：PNG, BMP, TIFF, WebP（推荐使用PNG以保持透明背景）
- 支持三种转换模式：
  - 默认模式：根据长边等比例缩放并相切
  - 四角相切模式（--fit）：图片完整显示，四角与圆形相切
  - 人脸检测模式（--face）：自动检测人脸并居中，图片填满圆形区域
- 支持多种尺寸单位：像素(px)、毫米(mm)、厘米(cm)
- 支持自定义DPI，影响最终输出分辨率
- 支持批量处理整个文件夹的图片
- 带有友好的命令行界面
- 自动选择输入输出文件夹
- 处理完成后自动打开输出文件夹

## 安装要求

1. Python 3.8 或更高版本
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

基本用法：
```bash
python circle_image_converter.py [选项]
```

### 命令行选项

- `--size`：设置圆形画布尺寸，支持以下格式：
  - 像素：500px（默认）
  - 毫米：50mm
  - 厘米：5cm
- `--dpi`：设置输出图片DPI，默认300
- `--fit`：启用四角相切模式
- `--face`：启用人脸检测和居中模式
- `--input-format`：指定输入图片格式，默认PNG
- `--output-format`：指定输出图片格式，默认PNG

### 使用示例

1. 使用默认设置（500px，PNG格式）：
```bash
python circle_image_converter.py
```

2. 指定尺寸为5厘米，DPI为300：
```bash
python circle_image_converter.py --size 5cm --dpi 300
```

3. 使用四角相切模式：
```bash
python circle_image_converter.py --fit
```

4. 使用人脸检测模式：
```bash
python circle_image_converter.py --face
```

5. 指定输出格式为WebP：
```bash
python circle_image_converter.py --output-format WEBP
```

### 输出说明

处理后的图片将根据使用的模式添加相应前缀：
- 默认模式：`default_`
- 四角相切模式：`fit_`
- 人脸检测模式：`face_`

例如：`photo.jpg` 在人脸检测模式下处理后将变为 `face_photo.png`

## 注意事项

1. 为保持透明背景，建议使用PNG格式输出
2. 人脸检测模式可能不会检测到某些图片中的人脸，此时将使用图片中心
3. 建议输入图片分辨率不要过小，以确保输出质量
4. 使用TIFF或PNG格式可以保持透明背景，其他格式将使用白色背景

## 技术支持

如有问题或建议，请提交Issue。
