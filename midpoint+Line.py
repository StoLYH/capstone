####### 중점 생성 코드 + 여백을 만들고 검은색으로 채우는 코드 #######

import json
import cv2
import matplotlib.pyplot as plt
import os
import math


# JSON 및 이미지 파일 경로 설정
json_folder = "617frame_updated"  # JSON 파일 폴더 경로
image_folder = "frames3_unet_767"  # 이미지 파일 폴더 경로
output_folder = "frames3_weldline123"  # 결과 저장 폴더

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 중점 계산 함수(json에 있는 좌표값으로 중점 계산)
def calculate_midpoints(corners):
    x_midpoints = []
    y_midpoints = []
    min_distance_midpoints = []
    used_indices = set()  # 이미 중점 만드는데 사용된 좌표

    for i in range(len(corners)):
        if i in used_indices:
            continue
        for j in range(i + 1, len(corners)):
            if j in used_indices:
                continue
            if corners[i][0] == corners[j][0]:
                midpoint = [corners[i][0], (corners[i][1] + corners[j][1]) / 2]
                x_midpoints.append(midpoint)
                used_indices.update([i, j])
                break
            elif corners[i][1] == corners[j][1]:
                midpoint = [(corners[i][0] + corners[j][0]) / 2, corners[i][1]]
                y_midpoints.append(midpoint)
                used_indices.update([i, j])
                break

    remaining_indices = [i for i in range(len(corners)) if i not in used_indices]
    for i in range(len(remaining_indices)):
        for j in range(i + 1, len(remaining_indices)):
            idx1, idx2 = remaining_indices[i], remaining_indices[j]
            distance = calculate_distance(corners[idx1], corners[idx2])
            midpoint = [
                (corners[idx1][0] + corners[idx2][0]) / 2,
                (corners[idx1][1] + corners[idx2][1]) / 2
            ]
            min_distance_midpoints.append((distance, midpoint))

    min_distance_midpoints.sort(key=lambda x: x[0])
    closest_midpoints = [midpoint for _, midpoint in min_distance_midpoints[:2]]

    return x_midpoints, y_midpoints, closest_midpoints

# 두 점 사이 거리 계산
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# 두 점 사이의 x,y 좌표 차이를 계산
def calculate_coordinate_difference(point1, point2):
    return abs(point1[0] - point2[0]), abs(point1[1] - point2[1])

# 처리 함수
def process_images(start_number, end_number):
    for number in range(start_number, end_number + 1):
        file_number = f"{number:04d}"
        json_file_path = os.path.join(json_folder, f"frame_{file_number}.json")
        image_file_path = os.path.join(image_folder, f"frame_{file_number}.png")

        if not os.path.exists(json_file_path) or not os.path.exists(image_file_path):
            print(f"파일 {file_number}에 해당하는 JSON 또는 이미지가 없습니다.")
            continue

        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            corners = [vertex for line in data['lines'] for vertex in line['vertices']]

            x_midpoints, y_midpoints, closest_midpoints = calculate_midpoints(corners)
            image = cv2.imread(image_file_path)
            if image is None:
                print(f"이미지를 불러오지 못했습니다: {image_file_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            fig = plt.figure(figsize=(8, 8), facecolor='black')
            ax = fig.add_subplot(111)
            ax.set_facecolor("black")
            ax.imshow(image)
            plt.axis("off")
            plt.xlim(0, 767)
            plt.ylim(767, 0)

            all_midpoints = x_midpoints + y_midpoints + closest_midpoints
            if len(all_midpoints) > 1:
                differences = [
                    (calculate_coordinate_difference(pt1, pt2), pt1, pt2)
                    for i, pt1 in enumerate(all_midpoints)
                    for pt2 in all_midpoints[i + 1:]
                ]

                differences.sort(key=lambda x: max(x[0]), reverse=True)
                connected_midpoints = set()

                for (x_diff, y_diff), point1, point2 in differences:
                    if tuple(point1) not in connected_midpoints and tuple(point2) not in connected_midpoints:
                        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'b-', linewidth=2)
                        connected_midpoints.update([tuple(point1), tuple(point2)])

                remaining_midpoints = [pt for pt in all_midpoints if tuple(pt) not in connected_midpoints]
                for i in range(len(remaining_midpoints) - 1):
                    point1 = remaining_midpoints[i]
                    point2 = remaining_midpoints[i + 1]
                    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'b-', linewidth=1)

            output_path = os.path.join(output_folder, f"{file_number}_midline.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=False)
            plt.close()

            print(f"파일 {file_number} 처리 완료: {output_path}")

        except FileNotFoundError:
            print(f"파일 {file_number}에 해당하는 JSON 또는 이미지를 찾을 수 없습니다.")
        except KeyError:
            print(f"JSON 파일 형식이 올바르지 않습니다: {json_file_path}")
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

start_number = int(input("시작 번호를 입력하세요: "))
end_number = int(input("끝 번호를 입력하세요: "))
process_images(start_number, end_number)






