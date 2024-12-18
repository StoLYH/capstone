import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from unet_model import UNet

# 모델을 이용해 이미지에 대해 segmentation을 수행하는 함수
def segment_image(model, image_tensor, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        weld_mask = outputs[0, 0, :, :].cpu().numpy()
        unprocessed_mask = outputs[0, 1, :, :].cpu().numpy()

        weld_mask_binary = (weld_mask > threshold).astype(np.uint8)
        unprocessed_mask_binary = (unprocessed_mask > threshold).astype(np.uint8)

    return weld_mask_binary, unprocessed_mask_binary

# 마스크를 원본 이미지 위에만 덧씌우는 함수
def overlay_masks_on_image(
    image, 
    weld_mask, 
    unprocessed_mask, 
    weld_color=(0, 255, 255),  
    unprocessed_color=(173, 216, 230),  # 전처리되지 않은 면 색상 (파란색)
    weld_alpha=0.8, 
    unprocessed_alpha=0.8
):
    """
    원본 이미지를 유지하면서 전처리되지 않은 면과 용접선에만 색을 입히는 함수.

    Args:
    - image: 원본 이미지 (PIL.Image 또는 numpy array)
    - weld_mask: 용접선 마스크 (numpy array, 0 또는 1 값)
    - unprocessed_mask: 전처리되지 않은 면 마스크 (numpy array, 0 또는 1 값)
    - weld_color: 용접선 색상 (tuple, RGB 값)
    - unprocessed_color: 전처리되지 않은 면 색상 (tuple, RGB 값)
    - weld_alpha: 용접선 투명도 (0~1)
    - unprocessed_alpha: 전처리되지 않은 면 투명도 (0~1)

    Returns:
    - overlayed_image: 색이 입혀진 이미지 (numpy array)
    """
    image_array = np.array(image).astype(np.float32)
    overlayed_image = image_array.copy()

    # 전처리되지 않은 면
    for c in range(3):
        overlayed_image[:, :, c] = np.where(
            unprocessed_mask > 0,
            (1 - unprocessed_alpha) * overlayed_image[:, :, c] + unprocessed_alpha * unprocessed_color[c],
            overlayed_image[:, :, c]
        )

    # 용접선
    for c in range(3):
        overlayed_image[:, :, c] = np.where(
            weld_mask > 0,
            (1 - weld_alpha) * overlayed_image[:, :, c] + weld_alpha * weld_color[c],
            overlayed_image[:, :, c]
        )

    return overlayed_image.astype(np.uint8)

if __name__ == "__main__":
    # 모델 경로 및 디렉토리 설정
    model_path = 'model_epoch_20.pth'
    current_dir = os.getcwd()
    frame_dir = os.path.join(current_dir, 'frames1')  # 프레임 이미지 디렉토리
    result_dir = os.path.join(current_dir, 'frames1_roo')  # 결과 저장 디렉토리
    os.makedirs(result_dir, exist_ok=True)  # 결과 디렉토리 생성

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = UNet(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 이미지 전처리 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))  # 모델 입력 크기로 리사이징
    ])

    # frames1 디렉토리에서 이미지 파일 리스트 불러오기
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png') or f.endswith('.jpg')])

    # 이미지 처리 및 결과 저장
    for i, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frame_dir, frame_file)  # 원본 이미지 경로
        save_path = os.path.join(result_dir, frame_file)  # 결과 저장 경로

        # 원본 이미지 로드 및 전처리
        image = Image.open(frame_path).convert("RGB")  # RGB 이미지로 변환
        image_tensor = transform(image)  # 텐서로 변환 및 리사이징

        # 마스크 예측
        weld_mask, unprocessed_mask = segment_image(model, image_tensor, device, threshold=0.5)

        # 마스크를 원본 이미지 위에 덧씌움
        overlayed_image = overlay_masks_on_image(image.resize((256, 256)), weld_mask, unprocessed_mask)

        # 결과 저장
        overlayed_image_pil = Image.fromarray(overlayed_image)
        overlayed_image_pil.save(save_path)  # 저장
        print(f"{i+1}/{len(frame_files)}: 결과 이미지가 {save_path} 경로에 저장되었습니다.")


# import torch
# import os
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from unet_model import UNet

# # 모델을 이용해 이미지에 대해 segmentation을 수행하는 함수
# def segment_image(model, image_tensor, device, threshold=0.5):
#     model.eval()
#     with torch.no_grad():
#         image_tensor = image_tensor.unsqueeze(0).to(device)
#         outputs = model(image_tensor)
#         weld_mask = outputs[0, 0, :, :].cpu().numpy()
#         unprocessed_mask = outputs[0, 1, :, :].cpu().numpy()

#         weld_mask_binary = (weld_mask > threshold).astype(np.uint8)
#         unprocessed_mask_binary = (unprocessed_mask > threshold).astype(np.uint8)

#     return weld_mask_binary, unprocessed_mask_binary

# # 마스크를 원본 이미지 위에만 덧씌우는 함수
# def overlay_masks_on_image(image, weld_mask, unprocessed_mask, weld_color=(144, 238, 144), unprocessed_color=(173, 216, 230), weld_alpha=0.8, unprocessed_alpha=0.8):
#     image_array = np.array(image).astype(np.float32)
#     overlayed_image = image_array.copy()

#     # 전처리되지 않은 면
#     for c in range(3):
#         overlayed_image[:, :, c] = np.where(
#             unprocessed_mask > 0,
#             (1 - unprocessed_alpha) * overlayed_image[:, :, c] + unprocessed_alpha * unprocessed_color[c],
#             overlayed_image[:, :, c]
#         )

#     # 용접선
#     for c in range(3):
#         overlayed_image[:, :, c] = np.where(
#             weld_mask > 0,
#             (1 - weld_alpha) * overlayed_image[:, :, c] + weld_alpha * weld_color[c],
#             overlayed_image[:, :, c]
#         )

#     return overlayed_image.astype(np.uint8)

# if __name__ == "__main__":
#     model_path = 'model_epoch_20.pth'
#     current_dir = os.getcwd()
#     frame_dir = os.path.join(current_dir, 'frames1')  # 프레임 이미지 디렉토리
#     result_dir = os.path.join(current_dir, 'frames1_result')  # 결과 저장 디렉토리
#     os.makedirs(result_dir, exist_ok=True)  # 결과 디렉토리 생성

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 모델 로드
#     model = UNet(in_channels=3, out_channels=2).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((256, 256))  # 모델 입력 크기
#     ])

#     # 동영상에서 추출된 프레임들을 처리
#     frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])  # 프레임 파일 리스트

#     for i, frame_file in enumerate(frame_files):  # 프레임별 처리
#         frame_path = os.path.join(frame_dir, frame_file)  # 프레임 이미지 경로
#         image = Image.open(frame_path).convert("RGB")  # 원본 이미지 로드
#         image_tensor = transform(image)  # 모델 입력 크기 맞추기 (리사이징 및 텐서 변환)

#         # 마스크 예측
#         weld_mask_pred, unprocessed_mask_pred = segment_image(model, image_tensor, device, threshold=0.5)

#         # 마스크를 원본 이미지 위에 덧씌움
#         overlayed_image = overlay_masks_on_image(image.resize((256, 256)), weld_mask_pred, unprocessed_mask_pred)

#         # 결과 저장
#         result_filename = f"result_{i:05d}.png"  # 저장 파일 이름 예: result_00000.png
#         save_path = os.path.join(result_dir, result_filename)
#         overlayed_image_pil = Image.fromarray(overlayed_image)
#         overlayed_image_pil.save(save_path)

#         print(f"{i+1}/{len(frame_files)}: 결과 이미지가 {save_path} 경로에 저장되었습니다.")
