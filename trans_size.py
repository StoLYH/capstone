import cv2
import os

# 입력 동영상 경로와 프레임 저장 디렉토리 설정
video_path = 'with_laser_straight.mp4'  # 입력 동영상 파일 이름
frame_dir = 'frames1'         # 프레임 저장 폴더
os.makedirs(frame_dir, exist_ok=True)

# 동영상 읽기
cap = cv2.VideoCapture(video_path)

# FPS와 총 프레임 수 계산
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 초당 프레임 수
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 수
duration = total_frames / fps  # 동영상 길이(초)

print(f"FPS: {fps}, 총 프레임: {total_frames}, 동영상 길이: {duration:.2f}초")

frame_count = 0

# 프레임 추출
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 저장할 프레임 경로 설정
    frame_path = os.path.join(frame_dir, f'frame_{frame_count:04d}.png')
    cv2.imwrite(frame_path, frame)  # 프레임 저장
    print(f"Saved: {frame_path}")
    frame_count += 1

cap.release()
print(f"총 {frame_count}개의 프레임이 저장되었습니다. (저장 경로: {frame_dir})")
