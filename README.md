# 圆形图像转换器 (Circle Image Converter)

一个简单易用的工具，可以将普通照片转换为圆形头像或圆形图片，支持多种处理模式和智能人脸检测。

## 功能特点

- 🔄 一键将方形图像转换为完美圆形
- 👤 智能人脸检测，自动居中人脸
- 🎨 可选背景颜色和透明度
- 📐 多种处理模式满足不同需求
- 🖥️ 简单直观的用户界面
- 🔍 高级自定义选项（尺寸、边框等）
- 📱 批量处理多个图片
- 🌍 中英文双语支持

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

### 直接运行可执行文件

1. 下载最新版本的可执行文件（exe）
2. 双击运行程序
3. 根据界面提示选择操作模式
4. 选择需要处理的图片
5. 查看结果并保存

### 命令行选项

程序支持多种命令行参数：

```
usage: circle_image_converter.py [-h] [-i INPUT] [-o OUTPUT] [-s SIZE] [-c COLOR] [-t TRANSPARENT] [-f] [-p] [-m] [--batch]

可选参数:
  -h, --help            显示帮助信息并退出
  -i INPUT, --input INPUT
                        输入图片路径或目录
  -o OUTPUT, --output OUTPUT
                        输出图片的路径或目录（默认为源文件名加_circle后缀）
  -s SIZE, --size SIZE  输出图片的尺寸，格式为WIDTHxHEIGHT（默认为原图尺寸）
  -c COLOR, --color COLOR
                        背景颜色，格式为HEX或RGB（默认为白色）
  -t TRANSPARENT, --transparent TRANSPARENT
                        设置背景透明度，0-100之间的整数（默认为0，完全不透明）
  -f, --fit             使用四角相切模式，较少裁剪
  -p, --face            使用人脸检测功能，自动将人脸居中
  -m, --manual          使用手动模式，可以自定义中心点
  --batch               批量处理整个目录
```

### 模式说明

1. **默认模式**：从图片中心裁剪最大内切圆
2. **四角相切模式**：保留图片更多内容，适合方形图片
3. **人脸检测模式**：自动识别并居中人脸
4. **人脸检测手动模式**：手动定位人脸位置

## 开发指南

### 环境配置

1. 克隆仓库：
```
git clone https://github.com/yourusername/Circle_images_converter.git
cd Circle_images_converter
```

2. 安装依赖：
```
pip install -r requirements.txt
```

### 从源码运行

```
python circle_image_converter.py
```

### 构建可执行文件

使用PyInstaller打包为独立可执行文件：

```
pyinstaller --onefile --icon=icon/app_icon.ico --add-data "haarcascades;haarcascades" --add-data "fonts;fonts" --hidden-import=cv2 --hidden-import=tkinter circle_image_converter.py
```

打包好的文件将在`dist`目录中。

## 常见问题

**Q: 程序无法启动怎么办？**  
A: 确保您的系统符合最低要求，Windows 7及以上，没有缺少必要的系统DLL文件。

**Q: 人脸检测不准确怎么办？**  
A: 尝试使用手动模式，或者调整图片的亮度和对比度后再次尝试。

**Q: 可以批量处理图片吗？**  
A: 是的，使用`--batch`参数或在高级选项中选择批量处理。

## 贡献

欢迎提交Pull Request或Issue。

## 许可证

MIT许可证