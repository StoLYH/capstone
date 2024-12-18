import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from vertex_predictor import get_model
import json
import os
from PIL import Image

# 데이터셋 클래스 정의
class VertexDataset(Dataset):
    def __init__(self, image_dir, label_file, json_dir, transform=None, target_size=(256, 256), original_size=(767, 767)):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.target_size = target_size
        self.original_size = original_size
        with open(label_file, 'r') as f:
            self.labels = json.load(f)
        
        # 용접선이 1개인 이미지만 필터링
        self.data = [img for img, weld_count in self.labels.items() if weld_count == 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)
        json_path = os.path.join(self.json_dir, img_name.replace('.png', '.json'))
        
        # 이미지 로드 및 변환
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # JSON 파일에서 꼭짓점 좌표 추출
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            vertices = json_data["lines"][0]["vertices"]
        
        # 원본 크기에서 target 크기로 좌표 스케일링
        scale_x = self.target_size[0] / self.original_size[0]
        scale_y = self.target_size[1] / self.original_size[1]
        scaled_vertices = [[int(x * scale_x), int(y * scale_y)] for x, y in vertices]

        # 꼭짓점 좌표를 하나의 벡터로 변환
        scaled_vertices = [coord for point in scaled_vertices for coord in point]
        return image, torch.tensor(scaled_vertices, dtype=torch.float32)

# 데이터 로더 설정
def get_dataloader(image_dir, label_file, json_dir, batch_size=16, target_size=(256, 256), original_size=(767, 767)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    dataset = VertexDataset(image_dir, label_file, json_dir, transform, target_size, original_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 학습 함수
def train_model(model, dataloader, epochs=10, learning_rate=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

# 메인 실행 부분
if __name__ == "__main__":
    image_dir = 'data15'           # 이미지 경로
    label_file = 'result.json'     # 용접선 개수가 기록된 파일 경로
    json_dir = 'data4'             # 정답 JSON 파일 경로
    
    # 데이터 로더 생성
    dataloader = get_dataloader(image_dir, label_file, json_dir, batch_size=16)
    
    # 모델 생성 및 학습
    model = get_model()
    train_model(model, dataloader, epochs=20, learning_rate=1e-4)
    
    # 학습 완료된 모델 저장
    torch.save(model.state_dict(), "vertex_predictor.pth")
    print("Model training completed and saved as vertex_predictor.pth")
