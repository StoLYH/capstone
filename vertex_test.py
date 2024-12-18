import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from vertex_predictor import get_model

# 모델 초기화 및 가중치 불러오기
model = get_model()
model.load_state_dict(torch.load("vertex_predictor.pth"))
model.eval()  # 모델을 평가 모드로 전환

# 이미지 전처리 함수
def preprocess_image(image_path, target_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(target_size),  # 모델 학습 시 사용한 이미지 크기로 리사이즈
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image  # 텐서와 원본 이미지를 함께 반환

# 예측 함수
def predict_coordinates(model, image_tensor):
    with torch.no_grad():  # 그래디언트 계산 비활성화 (평가 모드)
        output = model(image_tensor)  # 예측 수행
    coordinates = output.squeeze().tolist()  # 배치 차원 제거 후 리스트로 변환
    return [(coordinates[i], coordinates[i+1]) for i in range(0, len(coordinates), 2)]

# 예측 결과를 이미지에 시각화하여 저장
def visualize_and_save(image, coordinates, output_path="predicted_image_with_points.png", original_size=(767, 767), target_size=(256, 256)):
    # 스케일 변환 적용 (예측된 좌표를 원본 이미지 크기로 변환)
    scale_x = original_size[0] / target_size[0]
    scale_y = original_size[1] / target_size[1]
    adjusted_coordinates = [(x * scale_x, y * scale_y) for (x, y) in coordinates]
    
    # 좌표 시각화
    draw = ImageDraw.Draw(image)
    for (x, y) in adjusted_coordinates:
        # 빨간색 원으로 각 좌표를 표시
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red", outline="red")
    image.save(output_path)
    print(f"Predicted image saved to {output_path}")

# 예제 이미지 파일 경로
image_path = 'data15/00000.png'  # 예시 이미지 파일 경로 (실제 파일로 수정)

# 이미지 전처리 및 원본 이미지 로드
image_tensor, original_image = preprocess_image(image_path)

# 좌표 예측
predicted_coordinates = predict_coordinates(model, image_tensor)

# 시각화 및 저장
visualize_and_save(original_image, predicted_coordinates, output_path="predicted_image_with_points.png")
