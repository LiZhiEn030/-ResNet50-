import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import re  # 新增正则表达式处理
from albumentations import Compose, Normalize, Resize, HorizontalFlip, RandomBrightnessContrast, CoarseDropout
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# --------------------
# 增强版数据预处理
# --------------------


class AdvancedVehicleDataset(Dataset):
    def __init__(self, img_dir, xml_dir, transform=None, target_size=512, mode='train', test_size=0.2, seed=42):
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.transform = transform
        self.target_size = target_size
        self.mode = mode
        self.test_size = test_size
        self.seed = seed

        # 类别合并映射（过滤Sedan/SUV）
        self.class_mapping = {
            'Bus': 0,  # 客车
            'Microbus': 0,  # 客车
            'Minivan': 1,  # 货车
            'Truck': 1  # 货车
        }

        # 正则表达式匹配尺寸
        self.size_pattern = re.compile(r'$$\[(\d+)$$\]')  # 注意双重转义

        # 自动生成有效文件列表
        self._prepare_dataset()

        # 自动划分数据集
        self._split_dataset()

    def create_train_transform(target_size):
        return Compose([
            Resize(target_size, target_size),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.3),
            # 修正CoarseDropout参数（新版本API）
            CoarseDropout(
                max_holes=8,
                max_height_size=32,  # 新参数名
                max_width_size=32,
                min_holes=1,
                min_height_size=8,
                min_width_size=8,
                p=0.5
            ),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params={
            'format': 'albumentations',
            'label_fields': ['class_labels'],
            'min_area': 4,  # 添加边界框最小面积限制
            'min_visibility': 0.1
        })
    def _prepare_dataset(self):
        """预处理所有有效文件"""
        self.valid_files = []

        # 遍历所有XML文件
        for xml_file in os.listdir(self.xml_dir):
            if not xml_file.endswith('.xml'):
                continue

            xml_path = os.path.join(self.xml_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 检查是否存在有效目标
            has_valid_object = False
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name in self.class_mapping:
                    has_valid_object = True
                    break

            if has_valid_object:
                base_name = xml_file.split('.')[0]
                jpg_path = os.path.join(self.img_dir, f"{base_name}.jpg")
                if os.path.exists(jpg_path):
                    self.valid_files.append(base_name)

    def _split_dataset(self):
        """自动划分训练集/验证集"""
        # 先获取所有有效文件
        all_files = self.valid_files

        # 进行分层抽样（根据每个文件的主类别）
        labels = []
        for file in all_files:
            xml_path = os.path.join(self.xml_dir, f"{file}.xml")
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取该文件的主类别（第一个有效目标的类别）
            main_class = None
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name in self.class_mapping:
                    main_class = self.class_mapping[class_name]
                    break
            labels.append(main_class)

        # 划分数据集（保持类别平衡）
        train_files, val_files = train_test_split(
            all_files,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=labels
        )

        # 根据模式选择数据
        if self.mode == 'train':
            self.final_files = train_files
        else:
            self.final_files = val_files

    def __getitem__(self, idx):
        """ 关键方法：必须正确缩进在类内部 """
        base_name = self.final_files[idx]

        # 1. 读取图像
        img_path = os.path.join(self.img_dir, f"{base_name}.jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. 解析XML标注
        xml_path = os.path.join(self.xml_dir, f"{base_name}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像原始尺寸
        size = root.find('size')
        width = int(self.size_pattern.search(size.find('width').text).group(1))
        height = int(self.size_pattern.search(size.find('height').text).group(1))

        # 3. 处理多个目标
        boxes = []
        labels = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.class_mapping:
                continue

            # 解析边界框（归一化坐标）
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text) / width
            ymin = int(bndbox.find('ymin').text) / height
            xmax = int(bndbox.find('xmax').text) / width
            ymax = int(bndbox.find('ymax').text) / height

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_mapping[class_name])

        # 4. 应用数据增强
        if self.transform:
            transformed = self.transform(
                image=img,
                bboxes=boxes,
                class_labels=labels
            )
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']

        # 5. 转换为Tensor
        targets = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }

        return img, targets
    def __len__(self):
        return len(self.final_files)

    def count_classes(dataset):
        counts = {0: 0, 1: 0}
        for i in range(len(dataset)):  # 使用索引遍历避免迭代器问题
            try:
                _, targets = dataset[i]
                for label in targets['labels']:
                    counts[label.item()] += 1
            except Exception as e:
                print(f"处理样本 {i} 时出错: {str(e)}")
                continue
        return counts
# --------------------
# 自适应数据增强
# --------------------
def create_train_transform(target_size):
    return Compose([
        Resize(target_size, target_size),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.3),
        CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params={'format': 'albumentations', 'label_fields': ['class_labels']})


def create_val_transform(target_size):
    return Compose([
        Resize(target_size, target_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params={'format': 'albumentations', 'label_fields': ['class_labels']})


# --------------------
# 双任务模型（检测+分类）
# --------------------
class DualTaskModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 使用ResNet50作为骨干网络
        # 修正1：正确定义backbone
        backbone = models.resnet50(pretrained=True)  # 创建ResNet50实例

        # 修正2：正确截取网络层
        # children()返回所有子模块，[:-2]移除最后两个层（AvgPool和FC）
        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-2]  # 输出特征图尺寸为 [batch, 2048, H/32, W/32]
        )

        # 检测头
        self.detection_head = nn.Sequential(
            # 输入通道数2048对应ResNet50最后一层的输出
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),  # 保持空间维度
            nn.ReLU(inplace=True),
            # 输出通道数：4（bbox坐标） + num_classes（分类概率）
            nn.Conv2d(512, 4 + num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)

        # 检测输出
        detection_out = self.detection_head(features)

        # 分类输出
        classification_out = self.classification_head(features)

        return detection_out, classification_out


# --------------------
# 自定义
# --------------------
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # 平衡检测和分类损失的权重

    def forward(self, det_outputs, cls_outputs, det_targets, cls_targets):
        # 检测损失（Smooth L1 + CrossEntropy）
        det_boxes = det_outputs[:, :, :4]
        det_cls = det_outputs[:, :, 4:]

        box_loss = nn.SmoothL1Loss()(det_boxes, det_targets['boxes'])
        cls_loss = nn.CrossEntropyLoss()(det_cls.permute(0, 2, 1), det_targets['labels'])

        # 分类损失
        main_cls_loss = nn.CrossEntropyLoss()(cls_outputs, cls_targets)

        return self.alpha * (box_loss + cls_loss) + (1 - self.alpha) * main_cls_loss


# --------------------
# 智能训练器
# --------------------
class SmartTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)
        self.criterion = MultiTaskLoss(alpha=0.7)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for images, targets in self.train_loader:
            images = images.to(self.device)

            # 重组目标数据
            det_targets = {
                'boxes': torch.stack([t['boxes'] for t in targets]).to(self.device),
                'labels': torch.stack([t['labels'] for t in targets]).to(self.device)
            }

            # 主分类标签（取第一个目标的类别）
            cls_targets = torch.tensor([t['labels'][0] for t in targets]).to(self.device)

            self.optimizer.zero_grad()

            det_outputs, cls_outputs = self.model(images)
            loss = self.criterion(det_outputs, cls_outputs, det_targets, cls_targets)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)

                # 重组目标数据
                det_targets = {
                    'boxes': torch.stack([t['boxes'] for t in targets]).to(self.device),
                    'labels': torch.stack([t['labels'] for t in targets]).to(self.device)
                }

                cls_targets = torch.tensor([t['labels'][0] for t in targets]).to(self.device)

                det_outputs, cls_outputs = self.model(images)
                loss = self.criterion(det_outputs, cls_outputs, det_targets, cls_targets)

                total_loss += loss.item()

                # 计算分类准确率
                _, predicted = torch.max(cls_outputs, 1)
                correct += (predicted == cls_targets).sum().item()
                total += cls_targets.size(0)

        return total_loss / len(self.val_loader), correct / total


# --------------------
# 主程序
# --------------------
if __name__ == '__main__':
    # 配置参数
    config = {
        'img_dir': os.path.join('BITVehicle_Dataset', 'image'),  # 图片目录
        'xml_dir': os.path.join('BITVehicle_Dataset', 'annotations'),  # 标注目录
        'batch_size': 16,
        'target_size': 640,
        'num_epochs': 30,
        'test_size': 0.2,  # 验证集比例
        'seed': 42,  # 随机种子
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }

    # 创建数据集（自动划分）
    train_dataset = AdvancedVehicleDataset(
        img_dir=config['img_dir'],
        xml_dir=config['xml_dir'],
        transform=create_train_transform(config['target_size']),
        mode='train',
        test_size=config['test_size'],
        seed=config['seed']
    )

    val_dataset = AdvancedVehicleDataset(
        img_dir=config['img_dir'],
        xml_dir=config['xml_dir'],
        transform=create_val_transform(config['target_size']),
        mode='val',
        test_size=config['test_size'],
        seed=config['seed']
    )

    # 输出统计信息
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print("类别分布示例:")


    def count_classes(dataset):
        counts = {0: 0, 1: 0}
        for _, targets in dataset:
            for label in targets['labels']:
                counts[label.item()] += 1
        return counts


    train_counts = count_classes(train_dataset)
    val_counts = count_classes(val_dataset)
    print(f"训练集 - 客车: {train_counts[0]}, 货车: {train_counts[1]}")
    print(f"验证集 - 客车: {val_counts[0]}, 货车: {val_counts[1]}")