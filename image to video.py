##### 입력 이미지에서 검은색 픽셀 제외하고 영상 만듬
# 만약 입력 이미지가 767*767이 아니면 이미지 테두리에 검은색 여백이 생김, 
# 이건 중점선 만들 때 이미지 해상도 좋게 하려고 이미지 리사이징 안 해서 생김, 그래서 여백이 존재하는 경우 검은색 여백은 지우고 영상 생성


import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os
import re
import numpy as np


# 검은색 픽셀 제거 함수
def remove_black_margin(image):
    """
    이미지에서 검은색 여백을 제거합니다.
    :param image: 입력 이미지 (numpy array)
    :return: 검은색 여백이 제거된 이미지
    """
    # Convert to grayscale to detect black areas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours to detect bounding box around non-black areas
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # Crop the image to the bounding box
        return image[y:y+h, x:x+w]
    return image  # Return original image if no contours found

# 동영상을 생성하는 함수
def create_video_from_images(image_folder, output_video, start_num, end_num, exclude_nums=None, fps=2, frame_repeat=10):
    """
    검은색 여백 제거 후 이미지를 동영상으로 생성합니다.
    start_num: 사용할 이미지 파일 번호의 시작값
    end_num: 사용할 이미지 파일 번호의 끝값
    exclude_nums: 영상을 만들 때 제외할 이미지 번호 리스트
    image_folder: 입력 이미지가 저장된 폴더 경로
    output_video: 생성할 동영상 파일 경로 및 이름
    fps: 초당 프레임 속도
    frame_repeat: 각 프레임을 반복해서 추가할 횟수
    """
    # 입력 폴더에서 모든 파일을 가져와서 정렬
    all_images = sorted(glob.glob(f"{image_folder}/*.png"))
    
    # 파일 이름에서 숫자 부분을 추출하여 원하는 범위 필터링
    selected_images = [
        img for img in all_images 
        if re.search(r'\d+', os.path.basename(img)) and 
           start_num <= int(re.search(r'\d+', os.path.basename(img)).group()) <= end_num
    ]
    
    # 특정 이미지는 영상 만들 때 제외
    if exclude_nums:
        selected_images = [
            img for img in selected_images
            if int(re.search(r'\d+', os.path.basename(img)).group()) not in exclude_nums
        ]
    
    # 선택된 이미지가 없을 경우
    if not selected_images:
        print("선택된 이미지가 없습니다. 경로와 파일 이름을 확인하세요.")
        return
    
    # 첫 번째 이미지를 읽어 크기(height, width), 채널(layer) 정보를 가져옴
    frame = cv2.imread(selected_images[0])
    frame = remove_black_margin(frame)  # 검은색 여백 제거
    if frame is None:
        print("이미지 파일을 읽을 수 없습니다. 파일 형식을 확인하세요.")
        return
    
    height, width, layers = frame.shape
    
    # 동영상 파일 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))  # width, height: 동영상의 해상도
    
    # 선택된 이미지를 하나씩 읽어 동영상에 추가
    for image in selected_images:
        frame = cv2.imread(image)
        frame = remove_black_margin(frame)  # 검은색 여백 제거
        if frame is None:
            continue
        # 각 프레임을 반복 추가, 한 프레임의 지속시간을 늘릴려는 목적
        for _ in range(frame_repeat):  
            video.write(frame)
    
    # 동영상 파일 생성 완료
    video.release()
    print(f"동영상 파일이 생성되었습니다: {output_video}")
    
    visualize_video_with_matplotlib(output_video, fps)

# 동영상을 시각화하는 함수
def visualize_video_with_matplotlib(video_path, fps):
    # 만들어진 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("동영상을 열 수 없습니다. 파일 경로를 확인하세요.")
        return

    # 동영상 속성 파악, 프레임의 너비(width)과 높이(height), 프레임 간 대기시간(delay)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / fps)  

    # 첫 번째 프레임을 화면에 표시
    fig, ax = plt.subplots()
    ax.set_xticks([])  
    ax.set_yticks([])  
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  

    # 캔버스와 figure의 배경을 검은색으로 설정
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    # 프레임 읽기, ret: 프레임 읽기 성공 여부, frame: 첫 번째 프레임 데이터
    ret, frame = cap.read()
    if not ret:
        print("첫 번째 프레임을 읽을 수 없습니다.")
        cap.release()
        return

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    img = ax.imshow(frame, aspect='auto')  # 첫 번째 프레임을 표시

    # 프레임 업데이트 함수
    def update_frame(i): 
        ret, frame = cap.read()  # 다음 프레임 읽기
        if not ret:
            cap.release()
            plt.close(fig)  
            return img

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
        img.set_array(frame)  # 새로운 프레임으로 업데이트
        return img

    ani = animation.FuncAnimation(
        fig, update_frame, interval=delay, blit=False
    )

    # 블랙 배경 확실히 적용
    fig.tight_layout(pad=0)
    plt.show()


# 실행 설정
image_folder = "frames2_unet"  # 입력 이미지가 저장된 폴더 경로
output_video = "output_video_frame2_unet.mp4"  # 저장할 동영상 파일 이름
start_num = 0  # 시작 이미지 번호
end_num = 617 # 끝 이미지 번호
exclude_nums = []  # 제외할 이미지 번호 리스트
fps = 50 # 프레임 속도 (FPS)
frame_repeat = 1  # 동일 프레임 반복 횟수

create_video_from_images(image_folder, output_video, start_num, end_num, exclude_nums, fps, frame_repeat)




