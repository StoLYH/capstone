import os
import json
import math
import torch
from torchvision import transforms
from PIL import Image
from vertex_predictor import get_model  # 모델 정의 함수

# 경로 설정
image_dir = "frames1_mask"  # 이미지 폴더 경로
output_json_dir = "output_json"  # 결과 JSON 저장 폴더
model_path = "vertex_predictor.pth"  # 모델 파일 경로

# 결과 JSON 저장 폴더 생성
os.makedirs(output_json_dir, exist_ok=True)

# 이미지 변환 설정
target_size = (256, 256)  # 모델 입력 크기
transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor()
])

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()  # 모델 구조 정의
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 각도 계산 함수
def calculate_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle_radians = math.atan2(dy, dx)  # 라디안 단위 각도
    angle_degrees = math.degrees(angle_radians)  # 도 단위 각도
    return angle_radians, angle_degrees

# 이미지 처리 및 JSON 생성 함수
def process_images(image_dir, output_json_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    scale_x, scale_y = 768 / target_size[0], 768 / target_size[1]  # 원본 크기로 좌표 복원

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)

        # 이미지 로드 및 전처리
        image = Image.open(img_path).convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)

        # 모델 예측
        with torch.no_grad():
            output = model(input_image).squeeze(0).cpu().numpy()

        # 좌표 복원
        vertices = [
            (int(output[i] * scale_x), int(output[i + 1] * scale_y))
            for i in range(0, len(output), 2)
        ]

        # 각도 계산
        angle_radians, angle_degrees = calculate_angle(
            vertices[0][0], vertices[0][1], vertices[1][0], vertices[1][1]
        )

        # JSON 생성
        result = {
            "image": img_name,
            "lines": [
                {
                    "center": [
                        sum(v[0] for v in vertices) // 4,
                        sum(v[1] for v in vertices) // 4
                    ],
                    "angle_radians": angle_radians,
                    "angle_degrees": angle_degrees,
                    "vertices": vertices
                }
            ]
        }

        # JSON 파일 저장
        json_path = os.path.join(output_json_dir, img_name.replace('.png', '.json'))
        with open(json_path, 'w') as json_file:
            json.dump(result, json_file, indent=4)

        print(f"Processed: {img_name}, Saved: {json_path}")

# 실행
if __name__ == "__main__":
    process_images(image_dir, output_json_dir)
    print("All images processed and JSON files saved.")
