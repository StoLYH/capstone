import torch
import torch.nn as nn
import torchvision.models as models

class VertexPredictor(nn.Module):
    def __init__(self):
        super(VertexPredictor, self).__init__()
        # Pre-trained ResNet 모델을 사용하여 특징 추출
        self.base_model = models.resnet18(pretrained=True)
        # 출력 레이어를 꼭짓점 좌표 8개 (x, y) 예측용으로 수정
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 8)

    def forward(self, x):
        return self.base_model(x)

# 모델을 생성하는 함수
def get_model():
    return VertexPredictor()
