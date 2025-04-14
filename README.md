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
- 🎛️ 交互式菜单：直观的命令行界面，方便选择操作模式

## 处理模式示例

### 1. 默认模式--default
默认模式会保持图片比例，并使图片的长边与圆形相切。

![默认模式示例](https://github.com/KARHAO/Circle_images_converter/blob/main/examples/default_mode.png)

### 2. 四角相切模式--fit
四角相切模式会确保原图完整显示，并使图片的四个角与圆形相切。

![四角相切模式示例](https://github.com/KARHAO/Circle_images_converter/blob/main/examples/fit_mode.png)

### 3. 人脸检测模式--face
人脸检测模式会自动识别人脸位置并居中，适合处理头像。

![人脸检测模式示例](https://github.com/KARHAO/Circle_images_converter/blob/main/examples/face_mode.png)

## 使用方法

### 1. 安装PYTHON

确保您已安装Python 3.7或更高版本。

### 2. 安装依赖

```bash
# CMD运行
pip install -r /path/to/requirements.txt
```

**重要说明：**
- 本程序依赖于OpenCV 4.8.0.76版本，请不要安装其他版本以避免冲突
- 请勿安装opencv-python-headless版本，它不支持GUI功能，会导致手动模式无法使用
- 如果已安装多个版本的OpenCV，请先卸载所有版本后重新安装

```bash
# 如果遇到手动模式无法使用的问题，请先卸载所有OpenCV版本
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
# 然后安装指定版本
pip install opencv-python==4.8.0.76
```

### 3. 运行程序

直接运行程序将显示交互式菜单界面：

```bash
# CMD运行
python circle_image_converter.py
```

## 交互式菜单使用说明

最新版本增加了交互式菜单功能，程序启动后会显示欢迎信息和可用命令，更方便用户操作。

### 主菜单选项

启动程序后，会显示以下选项：
```
可用命令:
  1: 默认模式 - 保持图片比例，长边与圆形相切
  2: 四角相切模式 - 确保原图完整显示
  3: 人脸检测模式 - 自动检测人脸并居中
  4: 人脸检测手动模式 - 可以手动选择每张图片的处理方式
  5: 高级选项 - 自定义尺寸、DPI等参数
  h: 显示帮助信息
  q: 退出程序
```

### 使用方法

1. 输入对应的数字或字母选择功能
2. 选择功能后，会弹出文件夹选择对话框，分别选择输入和输出文件夹
3. 根据选择的功能执行相应的处理
4. 处理完成后返回主菜单，可以继续选择其他功能或退出

### 高级选项

选择`5`进入高级选项模式，可以自定义以下参数：
- 圆形画布尺寸（支持px、mm、cm单位）
- 输出DPI值
- 输入和输出图片格式
- 是否启用预览功能
- 是否添加模式前缀
- 处理模式（默认、四角相切、人脸检测、手动模式）

### 传统命令行参数

程序仍然支持传统的命令行参数方式运行：

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

### 高级命令行选项

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

手动模式是一个交互式功能，允许您为每张图片单独选择处理方式。

### 使用方法

1. 通过交互式菜单选择选项`4`或使用命令行启动手动模式：
   ```bash
   python circle_image_converter.py --face --manual
   ```

2. 程序会弹出一个预览窗口，显示当前图片和检测到的特征（如果有）

3. 使用以下键选择处理模式：
   - 按 `1` 键：选择人脸检测模式（根据检测到的人脸特征居中）
   - 按 `2` 键：选择默认模式（保持图片比例，长边与圆形相切）
   - 按 `3` 键：选择四角相切模式（确保原图完整显示）
   - 按 `ESC` 键：跳过当前图片（不处理）

4. 选择后，程序会自动处理当前图片并继续下一张

### 故障排除

如果手动模式窗口没有显示：

1. 确保安装了正确版本的OpenCV（4.8.0.76）
2. 确保没有安装不支持GUI的版本（如opencv-python-headless）
3. 在某些系统上，可能需要管理员权限运行程序
4. 检查是否有其他窗口遮挡了预览窗口

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

5. 在某些环境下，可能需要额外安装系统依赖才能正常使用GUI功能

6. 交互式菜单模式下，可以重复执行不同操作，更加方便灵活