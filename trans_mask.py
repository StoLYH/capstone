import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from unet_model import UNet

# 모델을 이용해 용접선 마스크를 추출하는 함수
def extract_weld_mask(model, image_tensor, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # 모델 입력 크기 (배치 추가)
        outputs = model(image_tensor)  # 모델 추론
        weld_mask = outputs[0, 0, :, :].cpu().numpy()  # 용접선 채널만 추출
        weld_mask_binary = (weld_mask > threshold).astype(np.uint8) * 255  # 이진화 마스크 생성 (0 또는 255)
    return weld_mask_binary

if __name__ == "__main__":
    # 경로 설정
    current_dir = os.getcwd()
    frames1_dir = os.path.join(current_dir, 'frames1')  # 입력 이미지 디렉토리
    frames1_mask_dir = os.path.join(current_dir, 'frames1_mask')  # 결과 마스크 저장 디렉토리
    os.makedirs(frames1_mask_dir, exist_ok=True)  # 결과 디렉토리 생성

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model_path = 'unet_final_model.pth'  # 학습된 모델 경로
    model = UNet(in_channels=3, out_channels=2).to(device)  # U-Net 모델 생성
    model.load_state_dict(torch.load(model_path, map_location=device))  # 모델 가중치 로드

    # 이미지 전처리 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))  # 모델 입력 크기로 리사이징
    ])

    # frames1 디렉토리에서 이미지 파일 리스트 불러오기
    frame_files = sorted([f for f in os.listdir(frames1_dir) if f.endswith('.png') or f.endswith('.jpg')])

    # 이미지 처리 및 결과 저장
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames1_dir, frame_file)  # 원본 이미지 경로
        mask_save_path = os.path.join(frames1_mask_dir, frame_file)  # 마스크 저장 경로

        # 원본 이미지 로드 및 전처리
        image = Image.open(frame_path).convert("RGB")  # RGB 이미지로 변환
        image_tensor = transform(image)  # 텐서로 변환 및 리사이징

        # 용접선 마스크 추출
        weld_mask = extract_weld_mask(model, image_tensor, device, threshold=0.5)

        # 마스크를 이미지로 저장
        weld_mask_image = Image.fromarray(weld_mask)  # numpy array -> PIL 이미지로 변환
        weld_mask_image.save(mask_save_path)  # 마스크 저장

        print(f"{i+1}/{len(frame_files)}: 용접선 마스크가 {mask_save_path}에 저장되었습니다.")
