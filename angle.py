import cv2 
import os
import json
import re
import glob

# JSON에서 각도 추출 함수
def extract_angle(json_file):
    """
    JSON 파일에서 첫 번째 'angle' 값을 소수점 4째자리로 반올림하여 반환.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
        angle = data['lines'][0]['angle']  # 첫 번째 라인의 각도
        return round(angle, 4)

# 텍스트를 이미지에 추가하는 함수
def add_angle_to_frame(image, angle, position=(10, 50), font_scale=0.5, color=(255, 0, 0), thickness=1):
    """
    이미지 오른쪽 상단에 검정색 글자만 추가합니다.
    """
    height, width = image.shape[:2]
    position = (width - 200 + 100, 40)  # 오른쪽 상단 위치
    
    # 검정색 텍스트 추가 (배경 없이)
    cv2.putText(
        image,
        f"Angle: {angle}",
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,  # 검정색 텍스트
        thickness,
        lineType=cv2.LINE_AA
    )
    return image

# 기존 동영상에 각도를 추가하는 함수
def add_angle_to_video(input_video, json_folder, output_video, fps=25):
    """
    기존 동영상에 JSON 파일의 각도 정보를 오른쪽 상단에 표시하는 새로운 동영상을 생성합니다.
    """
    # 기존 동영상 열기
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("동영상을 열 수 없습니다.")
        return

    # 동영상 속성 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 새 동영상 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # JSON 파일 정렬
    json_files = sorted(glob.glob(f"{json_folder}/*.json"))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(json_files):
            break

        # JSON 파일에서 각도 가져오기
        json_file = json_files[frame_idx]
        angle = extract_angle(json_file)

        # 각도를 프레임에 추가
        frame = add_angle_to_frame(frame, angle)

        # 새 동영상에 프레임 추가
        video_writer.write(frame)

        frame_idx += 1

    # 동영상 저장 및 해제
    cap.release()
    video_writer.release()
    print(f"새 동영상이 생성되었습니다: {output_video}")

# 실행
input_video = "out.mp4"  # 기존 동영상 파일
json_folder = "labeling_straight"  # JSON 파일 폴더
output_video = "out1.mp4"  # 새 동영상 파일 경로

add_angle_to_video(input_video, json_folder, output_video)
