import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from my_data_loader import WeldingDataset
from unet_model import UNet
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter  # 텐서보드 라이브러리

# 하이퍼파라미터 설정
learning_rate = 1e-4
batch_size = 8
num_epochs = 30
early_stop = 5 

# 현재 작업 디렉토리 경로 가져오기
current_dir = os.getcwd()
best_model_path = os.path.join(current_dir, '100bce_unet_model.pth')

# 텐서보드 설정
writer = SummaryWriter(log_dir=os.path.join(current_dir, 'runs'))

# 데이터셋 및 데이터로더
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

# 데이터 경로 설정
data1_dir = os.path.join(current_dir, 'data1')
data2_dir = os.path.join(current_dir, 'data2')
data3_dir = os.path.join(current_dir, 'data3')

# 데이터 리스트 생성
image_paths = sorted([f for f in os.listdir(data1_dir) if f.endswith('.png')])
mask_paths = sorted([f for f in os.listdir(data2_dir) if f.endswith('.png')])
unprocessed_paths = sorted([f for f in os.listdir(data3_dir) if f.endswith('.png')])

# 데이터를 train/val/test 세트로 나누기 (train: 8000, val: 1000, test: 1000)
train_real_imgs, temp_real_imgs, train_welding_mask, temp_welding_masks, train_unprocessed_mask, temp_unprocessed_mask = train_test_split(
    image_paths, mask_paths, unprocessed_paths, test_size=2000, random_state=42) # 8 : 2

val_real_imgs, test_real_imgs, val_welding_masks, test_welding_masks, val_unprocessed_mask, test_unprocessed_mask = train_test_split(
    temp_real_imgs, temp_welding_masks, temp_unprocessed_mask, test_size=1000, random_state=42) # 1 : 1

# 데이터셋 정의
train_dataset = WeldingDataset(data1_dir, data2_dir, data3_dir, transform, train_real_imgs, train_welding_mask, train_unprocessed_mask)
val_dataset = WeldingDataset(data1_dir, data2_dir, data3_dir, transform, val_real_imgs, val_welding_masks, val_unprocessed_mask)
test_dataset = WeldingDataset(data1_dir, data2_dir, data3_dir, transform, test_real_imgs, test_welding_masks, test_unprocessed_mask)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 생성
model = UNet(in_channels=3, out_channels=2).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 데이터 분리 결과를 JSON 파일로 저장
data_splits = {
    "train_real_imgs": train_real_imgs,
    "val_real_imgs": val_real_imgs,
    "test_real_imgs": test_real_imgs,
    "train_welding_masks": train_welding_mask,
    "val_welding_masks": val_welding_masks,
    "test_welding_masks": test_welding_masks,
    "train_unprocessed_mask": train_unprocessed_mask,
    "val_unprocessed_mask": val_unprocessed_mask,
    "test_unprocessed_mask": test_unprocessed_mask
}

with open('train_val_test_splits.json', 'w') as f:
    json.dump(data_splits, f)

# 훈련 및 검증 코드
best_val_loss = float('inf')
no_improvement_epochs = 0

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} start")
    model.train()
    epoch_loss = 0
    
    for batch_idx, (images, weld_masks, unprocessed_masks) in enumerate(train_loader):
        images, weld_masks, unprocessed_masks = images.to(device), weld_masks.to(device), unprocessed_masks.to(device)
        
        weld_masks = weld_masks.squeeze(1)
        unprocessed_masks = unprocessed_masks.squeeze(1)

        outputs = model(images)
        weld_output = outputs[:, 0, :, :]
        unprocessed_output = outputs[:, 1, :, :]

        loss_weld = criterion(weld_output, weld_masks)
        loss_unprocessed = criterion(unprocessed_output, unprocessed_masks)
        loss = loss_weld + loss_unprocessed

        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_train_loss = epoch_loss / len(train_loader)
    print(f'Epoch {epoch+1} completed, Average Train Loss: {avg_train_loss}')
    
    # 텐서보드에 훈련 손실 기록
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)

    # 검증 단계
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, weld_masks, unprocessed_masks in val_loader:
            images, weld_masks, unprocessed_masks = images.to(device), weld_masks.to(device), unprocessed_masks.to(device)
            
            weld_masks = weld_masks.squeeze(1)
            unprocessed_masks = unprocessed_masks.squeeze(1)

            outputs = model(images)
            weld_output = outputs[:, 0, :, :]
            unprocessed_output = outputs[:, 1, :, :]

            loss_weld = criterion(weld_output, weld_masks)
            loss_unprocessed = criterion(unprocessed_output, unprocessed_masks)
            val_loss += (loss_weld + loss_unprocessed).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss after Epoch {epoch+1}: {avg_val_loss}')
    
    # 텐서보드에 검증 손실 기록
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

    # 최적 모델 저장 및 조기 종료
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improvement_epochs = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved at Epoch {epoch+1}")
    else:
        no_improvement_epochs += 1

    if no_improvement_epochs >= early_stop:
        print(f"Early stopping at Epoch {epoch+1}")
        break

# 최종 최적 모델 불러오기
model.load_state_dict(torch.load(best_model_path))

# 텐서보드 종료
writer.close()



