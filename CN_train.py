import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from CN_welding_line_classifier import SimpleCNN
from CN_data_loader import WeldingDataset

# 하이퍼파라미터 설정
batch_size = 32
learning_rate = 1e-4
num_epochs = 20

# 데이터 경로
image_dir = 'data15'
json_dir = 'data4'

# 데이터셋 및 데이터로더 정의
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
train_dataset = WeldingDataset(image_dir, json_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 모델, 손실 함수, 옵티마이저 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device 설정
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # device를 직접 사용
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")

# 모델 저장
torch.save(model.state_dict(), 'CN_welding_line_classifier.pth')
