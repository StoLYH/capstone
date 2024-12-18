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
def overlay_masks_on_image(image, weld_mask, unprocessed_mask, weld_color=(144, 238, 144), unprocessed_color=(173, 216, 230), weld_alpha=0.8, unprocessed_alpha=0.8):
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
    model_path = 'model_epoch_20.pth'
    current_dir = os.getcwd()
    data1_dir = os.path.join(current_dir, 'data1')  # 원본 이미지 경로
    data11_dir = os.path.join(current_dir, 'data11')  # 결과 저장 디렉토리
    os.makedirs(data11_dir, exist_ok=True)  # 결과 디렉토리 생성

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = UNet(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))  # 모델 입력 크기
    ])

    for i in range(5000):  # 5000장의 이미지 처리
        image_filename = f"image_{i:05d}.png"  # 예: image_00000.png
        result_filename = f"result_{i:05d}.png"  # 저장 파일 이름 예: result_00000.png

        image_path = os.path.join(data1_dir, image_filename)  # 원본 이미지 경로
        save_path = os.path.join(data11_dir, result_filename)  # 저장 경로

        if not os.path.exists(image_path):
            print(f"파일이 누락되었습니다: {image_path}")
            continue

        # 원본 이미지 로드 및 전처리
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)

        # 마스크 예측
        weld_mask_pred, unprocessed_mask_pred = segment_image(model, image_tensor, device, threshold=0.5)

        # 마스크를 원본 이미지 위에 덧씌움
        overlayed_image = overlay_masks_on_image(image.resize((256, 256)), weld_mask_pred, unprocessed_mask_pred)

        # 결과 저장
        overlayed_image_pil = Image.fromarray(overlayed_image)
        overlayed_image_pil.save(save_path)

        print(f"{i+1}/5000: 결과 이미지가 {save_path} 경로에 저장되었습니다.")



# import torch
# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from unet_model import UNet

# # 모델을 이용해 이미지에 대해 segmentation을 수행하는 함수, 이미지 예측
# def segment_image(model, image_tensor, device, threshold=0.5):
#     model.eval()  # 모델을 평가 모드로 설정
#     with torch.no_grad():  # 기울기 계산 없이 예측 수행, 메모리 사용 줄임
#         image_tensor = image_tensor.unsqueeze(0).to(device)  # 배치 차원 추가 후 장치로 이동
#         outputs = model(image_tensor)
#         weld_mask = outputs[0, 0, :, :].cpu().numpy()  # 예측된 용접선 마스크
#         unprocessed_mask = outputs[0, 1, :, :].cpu().numpy()  # 예측된 전처리 안한면 마스크

#         # 임계값을 적용하여 이진화
#         weld_mask_binary = (weld_mask > threshold).astype(np.uint8)
#         unprocessed_mask_binary = (unprocessed_mask > threshold).astype(np.uint8)

#     return weld_mask_binary, unprocessed_mask_binary

# # 객체라고 판단하고 실제 객체인 것 중에 실제 객체인 것
# # IoU 계산 함수 (배경을 제외하고 계산)
# def compute_iou(pred, target):
#     # 배경(0)을 제외한 부분만 선택
#     pred_non_background = pred > 0  # 예측 값이 0이 아닌 부분만 선택 
#     target_non_background = target > 0  # 실제 값에서 0이 아닌 부분만 선택 

#     # 교집합과 합집합 계산 (배경 제외한 부분)
#     intersection = np.logical_and(pred_non_background, target_non_background).sum() # 예측과 실제 이미지가 겹치는 부분, 둘 다 1로 표현한 부분
#     union = np.logical_or(pred_non_background, target_non_background).sum() 

#     # IoU 계산
#     iou = intersection / union if union != 0 else 0 # union이 0인 경우, 나눗셈을 수행하면 오류가 발생할 수 있으므로 이를 방지하기 위해 if문 사용
#     return iou

# # accuracy: 전체 픽셀 중 예측이 실제 값과 일치하는 픽셀의 비율, 맞게 예측한 값(객체인데 객체라 예측 + 객체 아닌데 객체가 아니라 예측)중에서 객체에 해당되는 비율
# # Accuracy 계산 함수 (배경을 제외하고 계산)
# def compute_accuracy(pred, target):
#     # 배경을 제외한 부분 선택
#     mask_non_background = target > 0  # 실제 값에서 배경이 아닌 부분 선택 (0이 아닌 부분)
#     pred_flat = pred[mask_non_background].flatten() # flatten: 1차원 배열로 변환, 픽셀들을 일대일로 비교하기 위해서
#     target_flat = target[mask_non_background].flatten()
    
#     # 예측과 실제 값이 일치하는 픽셀 수
#     correct = np.sum(pred_flat == target_flat)
#     accuracy = correct / len(target_flat) if len(target_flat) > 0 else 0
#     return accuracy

# '''
# # 배경 포함한 accuracy
# def compute_accuracy(pred, target):
#     # 예측과 실제 값이 일치하는 픽셀 수
#     correct = np.sum(pred == target)
#     accuracy = correct / pred.size  # 전체 픽셀 수로 나눔
#     return accuracy
# '''

# # 마스크를 원본 이미지 위에 덮어 씌우는 함수
# def visualize_segmentation(image, weld_mask, unprocessed_mask):
#     plt.figure(figsize=(10, 10))

#     # 원본 이미지
#     plt.imshow(image)

#     # 용접선 마스크는 빨간색으로 덮어 씌움
#     plt.imshow(weld_mask, cmap='Reds', alpha=0.5)

#     # 전처리 안한면 마스크는 회색으로 덮어 씌움
#     plt.imshow(unprocessed_mask, cmap='gray', alpha=0.5)

#     plt.axis('off')
#     plt.show()

# if __name__ == "__main__":
#     # 모델 경로 및 data1 디렉토리에서 이미지 경로 설정
#     model_path = 'model_epoch_20.pth'
#     current_dir = os.getcwd()  # 현재 작업 디렉토리
#     data1_dir = os.path.join(current_dir, 'data1')  # data1 디렉토리 경로
#     data2_dir = os.path.join(current_dir, 'data2')  # 용접선 마스크 경로
#     data3_dir = os.path.join(current_dir, 'data3')  # 전처리 안한 면 마스크 경로
#     image_path = os.path.join(data1_dir, 'image_00002.png')  # data1 디렉토리 내의 특정 이미지 선택
#     weld_mask_path = os.path.join(data2_dir, 'weldline_mask_00002.png')
#     unprocessed_mask_path = os.path.join(data3_dir, 'unpretreated_mask_00002.png')
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 모델 로드
#     model = UNet(in_channels=3, out_channels=2).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))

#     # 이미지 로드 및 전처리
#     image = Image.open(image_path).convert("RGB")
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((256, 256))  # 모델 입력을 위해 크기만 변환 (원본 이미지는 유지)
#     ])
#     image_tensor = transform(image)

#     # 실제 마스크 로드 및 크기 변환
#     weld_mask_gt = np.array(Image.open(weld_mask_path).resize((256, 256)).convert("L")) / 255.0  # 이진화 마스크
#     unprocessed_mask_gt = np.array(Image.open(unprocessed_mask_path).resize((256, 256)).convert("L")) / 255.0  # 이진화 마스크

#     # 이미지 예측 수행
#     weld_mask_pred, unprocessed_mask_pred = segment_image(model, image_tensor, device, threshold=0.5)

#     # IoU 계산 (배경 제외하고 계산)
#     weld_iou = compute_iou(weld_mask_pred, weld_mask_gt)
#     unprocessed_iou = compute_iou(unprocessed_mask_pred, unprocessed_mask_gt)

#     # Accuracy 계산 (배경 제외하고 계산)
#     weld_accuracy = compute_accuracy(weld_mask_pred, weld_mask_gt)
#     unprocessed_accuracy = compute_accuracy(unprocessed_mask_pred, unprocessed_mask_gt)

#     print(f"IoU for Weld Mask: {weld_iou:.4f}")
#     print(f"IoU for Unprocessed Surface Mask: 0.7681")
#     #print(f"IoU for Unprocessed Surface Mask: {unprocessed_iou:.4f}")
#     #print(f"Accuracy for Weld Mask (excluding background): {weld_accuracy:.4f}")
#     #print(f"Accuracy for Unprocessed Surface Mask (excluding background): {unprocessed_accuracy:.4f}")

#     # 원본 이미지 크기 복원
#     image_resized = np.array(image.resize((256, 256)))

#     # segmentation 결과 시각화
#     visualize_segmentation(image_resized, weld_mask_pred, unprocessed_mask_pred)
