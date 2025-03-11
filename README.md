# 圆形图片转换工具

一个功能强大的图片处理工具，可以将各种格式的图片转换为带透明背景的圆形图片。支持智能人脸检测和居中，适用于制作头像、徽标等场景。

## 功能特点

- **多种处理模式**
  - 默认模式：根据长边等比例缩放并相切
  - 四角相切模式：确保原图完整显示且四角与圆形相切
  - 人脸检测模式：智能检测人脸位置并居中

- **丰富的图片格式支持**
  - 支持输入：PNG, JPG/JPEG, BMP, GIF, WebP, TIFF, JFIF
  - 支持输出：PNG（推荐，支持透明背景）, JPEG, BMP, WebP, TIFF, JFIF

- **灵活的尺寸设置**
  - 支持像素(px)、毫米(mm)、厘米(cm)多种单位
  - 可自定义DPI，实现高质量输出

- **智能特征检测**
  - 正面人脸检测
  - 侧面人脸检测
  - 上半身躯干检测
  - 眼睛检测

- **用户友好界面**
  - 图形化文件夹选择
  - 处理进度实时显示
  - 可选的预览窗口
  - 详细的处理结果统计

## 安装说明

1. 确保已安装 Python 3.6 或更高版本
2. 克隆或下载本项目
3. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 基本用法

```bash
# 使用默认设置（500px，PNG格式）
python circle_image_converter.py

# 指定输出尺寸为1000像素，使用四角相切模式
python circle_image_converter.py --size 1000px --fit

# 使用人脸检测模式，输出为300DPI的JPEG格式，并启用预览窗口
python circle_image_converter.py --face --dpi 300 --output-format JPEG --preview
```

### 命令行参数

- `--size`：设置圆形画布尺寸，支持 px/mm/cm 单位（默认：500px）
- `--dpi`：设置输出图片DPI，影响最终分辨率（默认：300）
- `--fit`：启用四角相切模式
- `--face`：启用人脸检测模式
- `--input-format`：指定输入图片格式（默认：PNG）
- `--output-format`：指定输出图片格式（默认：PNG）
- `--preview`：启用处理过程预览窗口

### 使用示例

1. **制作头像**
   ```bash
   python circle_image_converter.py --face --size 500px --output-format PNG
   ```

2. **制作高清徽标**
   ```bash
   python circle_image_converter.py --fit --size 1000px --dpi 600
   ```

3. **批量处理带预览**
   ```bash
   python circle_image_converter.py --face --preview
   ```

## 注意事项

1. 人脸检测模式下，如果未检测到人脸，将自动使用图像中心作为参考点
2. 建议使用PNG格式作为输出格式，以保持透明背景
3. 处理大量图片时，建议关闭预览窗口以提高处理速度

## 常见问题

Q: 为什么有些图片无法检测到人脸？
A: 人脸检测的准确性受多个因素影响，如光线、角度、清晰度等。如果未检测到人脸，程序会自动使用默认的居中处理。

Q: 如何选择合适的输出尺寸？
A: 建议根据实际用途选择：
- 网站头像：300-500px
- 打印用图：建议使用mm/cm单位，并适当提高DPI

## 许可证

MIT License
