import os
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50

# Создаем папку для сохранения результатов, если ее нет
output_dir = Path('./output_data')
output_dir.mkdir(exist_ok=True)

DATA_DIR = Path('./CamVid')

if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
    url = "https://github.com/alexgkendall/SegNet-Tutorial/archive/master.zip"
    urllib.request.urlretrieve(url, "camvid.zip")
    with zipfile.ZipFile("camvid.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    (Path('./SegNet-Tutorial-master/CamVid')).rename(DATA_DIR)
    os.remove("camvid.zip")

IMG_DIR = DATA_DIR / 'train'
MASK_DIR = DATA_DIR / 'trainannot'

image_paths = sorted(list(IMG_DIR.glob('*.png')))
mask_paths = sorted(list(MASK_DIR.glob('*.png')))

print(f"Images found: {len(image_paths)}")
print(f"Masks found: {len(mask_paths)}")

assert len(image_paths) > 0, "Изображения не найдены"
assert len(mask_paths) > 0, "Маски не найдены"
assert len(image_paths) == len(mask_paths), "Количество изображений и масок не совпадает"

image_paths = image_paths[:1000]
mask_paths = mask_paths[:1000]


class CamVidDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.num_classes = 12

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].long()
        else:
            image = TF.to_tensor(image)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

# Аугментации для тренировочных данных включают:
# - RandomScale: случайное масштабирование
# - HorizontalFlip: горизонтальное отражение
# - RandomBrightnessContrast: изменение яркости и контраста
# - Affine: аффинные преобразования (сдвиг, масштаб, поворот)
# Эти преобразования помогают модели лучше обобщать и избегать переобучения
train_transform = A.Compose([
    A.RandomScale(scale_limit=0.5, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=15, p=0.5),
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_size = int(0.2 * len(image_paths))
train_size = len(image_paths) - val_size

train_image_paths = image_paths[:train_size]
train_mask_paths = mask_paths[:train_size]
val_image_paths = image_paths[train_size:]
val_mask_paths = mask_paths[train_size:]

train_dataset = CamVidDataset(train_image_paths, train_mask_paths, transform=train_transform)
val_dataset = CamVidDataset(val_image_paths, val_mask_paths, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

# Используется модель DeepLabV3 с ResNet50 в качестве backbone
# Выбор обоснован тем, что DeepLabV3 показывает хорошие результаты в задачах семантической сегментации
# благодаря использованию атрофированных сверток (dilated convolutions) и ASPP модулю
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 12, kernel_size=1)
model.aux_classifier[4] = nn.Conv2d(256, 12, kernel_size=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Используется кросс-энтропийная функция потерь, подходящая для задач классификации
criterion = nn.CrossEntropyLoss()
# Оптимизатор AdamW с learning rate 3e-4 (типичное значение для подобных задач)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
# Планировщик learning rate с уменьшением в 10 раз каждые 10 эпох
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#  Функция для подсчета mean IoU
def mean_iou(preds, labels, num_classes=12):
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(ious)

#  Обучение на одну эпоху
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

# Основная метрика - mean Intersection over Union (mIoU)
# Показывает насколько хорошо совпадают предсказанные и истинные сегменты
def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(1)
            total_iou += mean_iou(preds, masks)
    return running_loss / len(loader.dataset), total_iou / len(loader)

# Обучение
num_epochs = 30
best_iou = 0

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_iou = eval_model(model, val_loader, criterion, device)
    scheduler.step()

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), 'best_deeplabv3_camvid.pth')

    print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}")

# Функция для визуального сравнения исходного изображения,
# истинной маски и предсказания модели
def visualize_and_save_sample(model, dataset, device, idx=0, epoch=None):
    model.eval()
    image, mask = dataset[idx]
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))['out']
        pred = output.argmax(1).squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)

    ax[0].imshow(img_np)
    ax[0].set_title('Image')
    ax[0].axis('off')

    ax[1].imshow(mask.cpu().numpy())
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')

    ax[2].imshow(pred)
    ax[2].set_title('Prediction')
    ax[2].axis('off')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if epoch is not None:
        filename = f"sample_epoch_{epoch}_{timestamp}.png"
    else:
        filename = f"sample_{timestamp}.png"

    save_path = output_dir / filename
    plt.show()


visualize_and_save_sample(model, val_dataset, device, idx=10)