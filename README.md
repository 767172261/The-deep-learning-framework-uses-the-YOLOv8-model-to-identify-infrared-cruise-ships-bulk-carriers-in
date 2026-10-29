深度学习框架使用 YOLOv8 模型对红外船舶数据集 通过训练的权重推理识别检测红外状态下的红外邮轮、散货船、红外军舰、帆船、皮划艇 、集装箱船、渔船目标。红外船只数据集的训练及应用。
@[toc]
以下文字及代码仅供参考学习使用。


**数据集描述：**
红外船舶数据集，8002张jpg图片，txt和xml标注格式。
7类船数据集，8002张，txt和xml标注。含不同场景下、时段、分辨率海上、港口和
海边邮轮数据集、散货船数据集、军舰数据集、帆船数据集、皮划艇数据集、集装箱船数据集、渔船目标数据集

该数据库记录包含了多个不同场景下、不同时段、不同分辨率的海上、港口和海边上的邮轮、散货船、军舰、帆船、皮划艇 、集装箱船、渔船目标。图像分辨率分别为：384*288、640*512、1280*1024，对图像中的七类船舶目标liner、bulk carrier、warship、sailboat、canoe、container ship、fishing boat进行了标注，该数据库将其主要用于真实世界红外海防领域的目标检测识别技术研究。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6ed00e7b5eaa4bb2887628ab0f267b5a.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d7b01cc422f3453292be97cbfb7e8ebe.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d88750a4add04f99b8920f2927b7d0f9.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/33136ae642cf4d8097d7de1e589b494c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/32281ce5446f4d69b16e7c3284dfbfa3.png)
使用 YOLOv8 对红外船舶数据集进行训练、推理和评估，按照以下步骤操作：

### 1. 环境准备

#### 安装 CUDA 驱动
确保系统已经安装了适合你 GPU 的 NVIDIA CUDA 驱动程序。你可以通过运行 `nvidia-smi` 命令来检查是否已正确安装。

#### 安装 Anaconda 和创建虚拟环境
首先，下载并安装 [Anaconda](https://www.anaconda.com/products/distribution)。然后创建一个新的 Python 虚拟环境：

```bash
conda create -n ship_detection python=3.8
conda activate ship_detection
```

#### 安装依赖项
在激活的虚拟环境中安装必要的库：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install ultralytics opencv-python matplotlib tqdm
```
请根据你的CUDA版本调整PyTorch的安装链接。

### 2. 数据集组织

确保数据集按如下结构组织（YOLOv8支持YAML格式的数据配置）：

```
ship_dataset/
├── images/
│   ├── train/
│   ├── val/
├── labels/
│   ├── train/
│   ├── val/
└── data.yaml
```

其中 `data.yaml` 文件应包含如下信息：

```yaml
train: ./ship_dataset/images/train
val: ./ship_dataset/images/val

nc: 7  # 类别数量
names: ['liner', 'bulk carrier', 'warship', 'sailboat', 'canoe', 'container ship', 'fishing boat']
```

标注文件是XML格式，转换为YOLO所需的TXT格式。使用脚本或工具进行转换。

### 3. 使用YOLOv8训练模型

#### 加载预训练模型

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # 可以选择其他尺寸的预训练模型
```

#### 训练模型

编写一个简单的Python脚本来启动训练过程：

```python
from ultralytics import YOLO

def train_yolo():
    model = YOLO('yolov8n.yaml')  # 如果要从头开始训练，可以指定yaml配置文件
    results = model.train(data='path/to/ship_dataset/data.yaml', epochs=100, imgsz=640)
    
if __name__ == '__main__':
    train_yolo()
```

确保将 `'path/to/ship_dataset/data.yaml'` 替换为实际路径。

### 4. 推理代码

用于对新的图像或视频进行检测：

```python
from ultralytics import YOLO
import cv2

def detect_yolo(image_path):
    model = YOLO('runs/detect/exp/weights/best.pt')  # 替换为你的最佳权重路径
    results = model.predict(source=image_path, save=True)

if __name__ == '__main__':
    detect_yolo('path/to/test/image.jpg')
```

### 5. 评估代码

使用YOLOv8内置的评估函数直接对验证集进行评估：

```python
from ultralytics import YOLO

def evaluate_yolo():
    model = YOLO('runs/detect/exp/weights/best.pt')  # 替换为你的最佳权重路径
    metrics = model.val()  # 自动对验证集进行评估
    
if __name__ == '__main__':
    evaluate_yolo()
```


