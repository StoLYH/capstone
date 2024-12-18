import sys
import os
import json
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, 
                             QVBoxLayout, QWidget, QHBoxLayout, QListWidget, QMessageBox,
                             QListWidgetItem, QFrame)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QFont, QKeySequence
from PyQt5.QtCore import Qt, QPoint, QEvent, QSize, QSettings
from PyQt5.QtWidgets import QShortcut

class LabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_size = QSize(768, 768)
        self.initUI()
        self.image_paths = []
        self.current_image_index = -1
        self.points = []  # 사용자가 찍은 점들을 저장하는 리스트
        self.midpoints = []  # 선분의 중간점들을 저장하는 리스트
        self.lines = []  # 완성된 선들을 저장하는 리스트
        self.cursor_pos = QPoint()
        self.dragging_point = None
        self.drag_threshold = 10  # 드래그 시작을 위한 임계값 설정 (픽셀 단위)
        self.labeled_images = set()  # 라벨링이 완료된 이미지들의 집합
        
        # 설정 로드
        self.settings = QSettings("YourCompany", "LabelingTool")
        self.labeling_dir = self.settings.value("labeling_dir", "")

    def initUI(self):
        # UI 초기화 및 윈도우 설정
        self.setWindowTitle('Line Labeling Tool')
        self.setGeometry(100, 100, 1000, 500)  # 윈도우 크기와 위치 설정

        main_layout = QHBoxLayout()
        
        # 이미지를 표시할 레이블 생성
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.image_size)  # 이미지 레이블의 크기 고정
        self.image_label.setAlignment(Qt.AlignCenter)  # 이미지를 중앙 정렬
        self.image_label.setMouseTracking(True)  # 마우스 추적 활성화
        self.image_label.installEventFilter(self)  # 이벤트 필터 설치
        main_layout.addWidget(self.image_label, 2)  # 이미지 레이블을 메인 레이아웃에 추가

        # 컨트롤 영역 (버튼, 리스트 등) 설정
        control_layout = QVBoxLayout()  # 컨트롤 영역을 위한 수직 박스 레이아웃 생성
        
        button_width = 180  # 버튼의 너비 설정
        
        # 각종 버튼 생성 및 레이아웃에 추가
        load_button = QPushButton('Load Images', self)
        load_button.clicked.connect(self.load_images)
        load_button.setFixedWidth(button_width)
        control_layout.addWidget(load_button)

        save_button = QPushButton('Save Labels (Ctrl+S)', self)
        save_button.clicked.connect(self.save_labels)
        save_button.setFixedWidth(button_width)
        control_layout.addWidget(save_button)

        change_dir_button = QPushButton('Change Save Directory', self)
        change_dir_button.clicked.connect(self.change_save_directory)
        change_dir_button.setFixedWidth(button_width)
        control_layout.addWidget(change_dir_button)

        clear_button = QPushButton('Clear All (Esc)', self)
        clear_button.clicked.connect(self.clear_all)
        clear_button.setFixedWidth(button_width)
        control_layout.addWidget(clear_button)

        undo_button = QPushButton('Undo Last Point (Ctrl+Z)', self)
        undo_button.clicked.connect(self.undo_last_point)
        undo_button.setFixedWidth(button_width)
        control_layout.addWidget(undo_button)

        self.next_button = QPushButton('Next Image', self)
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setFixedWidth(button_width)
        control_layout.addWidget(self.next_button)

        self.prev_button = QPushButton('Previous Image', self)
        self.prev_button.clicked.connect(self.prev_image)
        self.prev_button.setFixedWidth(button_width)
        control_layout.addWidget(self.prev_button)

        # 정보 표시 레이블 생성
        self.info_label = QLabel('Click to add points.', self)
        control_layout.addWidget(self.info_label)

        # 좌표 표시 레이블 생성
        self.coord_label = QLabel('Coordinates: ', self)
        control_layout.addWidget(self.coord_label)

        # 이미지 리스트 위젯 생성
        self.image_list = QListWidget(self)
        self.image_list.setFixedSize(button_width, 450)  # 리스트 위젯의 크기 고정
        self.image_list.itemClicked.connect(self.select_image_from_list)
        control_layout.addWidget(self.image_list)

        # 라벨링 진행 상황 표시 레이블 생성
        self.progress_label = QLabel('Labeled: 0 / 0', self)
        control_layout.addWidget(self.progress_label)

        control_layout.addStretch(1)

        # 컨트롤 프레임 생성
        control_frame = QFrame()
        control_frame.setLayout(control_layout)
        control_frame.setFrameShape(QFrame.StyledPanel)  # 프레임에 스타일 적용
        
        main_layout.addWidget(control_frame, 1)  # 컨트롤 프레임을 메인 레이아웃에 추가

        # 중앙 위젯 설정
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 키보드 단축키 설정
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_last_point)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_labels)
        QShortcut(QKeySequence("Esc"), self, self.clear_all)

    def change_save_directory(self):
        """저장 디렉토리를 변경하고 라벨링 상태를 업데이트하는 메서드"""
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        new_dir = QFileDialog.getExistingDirectory(self, "Select New Save Directory", options=options)
        if new_dir:
            self.labeling_dir = new_dir
            self.settings.setValue("labeling_dir", self.labeling_dir)
            self.info_label.setText(f'Save directory changed to: {self.labeling_dir}')
            
            # 라벨링 상태 재검사
            self.labeled_images.clear()
            for i, path in enumerate(self.image_paths):
                if self.check_if_labeled(path):
                    self.labeled_images.add(path)
                    item = self.image_list.item(i)
                    item.setText(f"[완료] {os.path.basename(path)}")
                else:
                    item = self.image_list.item(i)
                    item.setText(os.path.basename(path))
            
            self.update_progress_label()

    def load_images(self):
        """이미지 파일들을 로드하고 리스트에 추가하는 메서드"""
        options = QFileDialog.Options()
        file_names, _ = QFileDialog.getOpenFileNames(self, "Load Images", "", 
                                                     "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_names:
            self.image_paths = file_names
            self.image_list.clear()
            for path in self.image_paths:
                item = QListWidgetItem(os.path.basename(path))
                if self.check_if_labeled(path):
                    item.setText(f"[완료] {os.path.basename(path)}")
                    self.labeled_images.add(path)
                self.image_list.addItem(item)
            self.current_image_index = 0
            self.load_current_image()
            self.update_progress_label()  # 진행 상황 업데이트

    def load_current_image(self):
        """현재 선택된 이미지를 로드하고 표시하는 메서드"""
        if 0 <= self.current_image_index < len(self.image_paths):
            self.current_image = QPixmap(self.image_paths[self.current_image_index])  # 현재 이미지를 QPixmap으로 로드
            self.points = []
            self.midpoints = []
            self.lines = []
            self.temp_line = None  # 임시 선 초기화
            self.image_list.setCurrentRow(self.current_image_index)  # 이미지 리스트에서 현재 이미지를 선택
            self.info_label.setText(f'Image {self.current_image_index + 1}/{len(self.image_paths)}')  # 정보 레이블을 업데이트
            self.update()

    def next_image(self):
        """다음 이미지로 이동하는 메서드"""
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.load_current_image()

    def prev_image(self):
        """이전 이미지로 이동하는 메서드"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()

    def select_image_from_list(self, item):
        """리스트에서 선택된 이미지로 이동하는 메서드"""
        self.current_image_index = self.image_list.row(item)
        self.load_current_image()

    def eventFilter(self, source, event):
        """마우스 이벤트를 필터링하는 메서드"""
        if source == self.image_label and event.type() == QEvent.MouseMove:
            pos = event.pos()
            self.update_cursor_position(pos)  # 커서 위치 업데이트
        return super().eventFilter(source, event)
            
    def mouseMoveEvent(self, event):
        """마우스 이동 이벤트를 처리하는 메서드"""
        if self.dragging_point is not None:
            pos = event.pos() - self.image_label.pos()  # 이미지 레이블 내에서의 상대적 위치 계산
            if 0 <= pos.x() < self.image_label.width() and 0 <= pos.y() < self.image_label.height():
                self.points[self.dragging_point] = pos  # 드래그 중인 점의 위치 업데이트
                self.update_lines_and_midpoints()  # 선과 중간점 업데이트
        elif len(self.points) % 2 == 1:
            self.temp_line = (self.points[-1], self.cursor_pos)  # 임시 선 업데이트
        self.update()

    def mousePressEvent(self, event):
        """마우스 클릭 이벤트를 처리하는 메서드"""
        if self.image_label.underMouse() and event.button() == Qt.LeftButton:
            pos = event.pos() - self.image_label.pos()
            if 0 <= pos.x() < self.image_label.width() and 0 <= pos.y() < self.image_label.height():
                for i, point in enumerate(self.points):
                    if (pos - point).manhattanLength() <= self.drag_threshold:
                        self.dragging_point = i  # 드래그할 점 설정
                        break
                else:
                    self.points.append(pos)  # 새로운 점 추가
                    self.update_lines_and_midpoints()  # 선과 중간점 업데이트
                self.update()
                self.info_label.setText(f'Points: {len(self.points)}, Midpoints: {len(self.midpoints)}')  # 정보 레이블 업데이트

    def mouseReleaseEvent(self, event):
        """마우스 릴리즈 이벤트를 처리하는 메서드"""
        if event.button() == Qt.LeftButton:
            self.dragging_point = None  # 드래그 상태 해제
            self.update()

    def update_cursor_position(self, pos):
        """커서 위치를 업데이트하고 표시하는 메서드"""
        if self.image_label.rect().contains(pos):
            self.cursor_pos = pos
            self.coord_label.setText(f'Coordinates: ({pos.x()}, {pos.y()})')
            self.update()

    def paintEvent(self, event):
        """이미지와 라벨링 요소들을 그리는 메서드"""
        super().paintEvent(event)
        if hasattr(self, 'current_image'):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            x = (self.image_label.width() - self.current_image.width()) // 2
            y = (self.image_label.height() - self.current_image.height()) // 2

            # 현재 이미지 그리기
            painter.drawPixmap(self.image_label.x() + x, self.image_label.y() + y, self.current_image)

            painter.translate(self.image_label.x() + x, self.image_label.y() + y)

            # 꼭짓점(빨간색) 그리기
            painter.setPen(QPen(QColor(255, 0, 0), 1))
            painter.setBrush(QBrush(QColor(255, 0, 0, 200)))
            for point in self.points:
                painter.drawEllipse(point, 3, 3)

            # 중간점(초록색) 그리기
            painter.setPen(QPen(QColor(0, 255, 0), 1))
            painter.setBrush(QBrush(QColor(0, 255, 0, 200)))
            for point in self.midpoints:
                painter.drawEllipse(point, 3, 3)

            # 파란색 선 그리기
            painter.setPen(QPen(QColor(0, 0, 255), 2))
            for line in self.lines:
                painter.drawLine(line[0], line[1])

            painter.resetTransform()

            # 커서 위치에 좌표 표시
            if self.image_label.rect().contains(self.cursor_pos):
                painter.setPen(QPen(Qt.white, 2))
                painter.setFont(QFont("Arial", 10, QFont.Bold))
                text = f"({self.cursor_pos.x()}, {self.cursor_pos.y()})"
                text_width = painter.fontMetrics().width(text)
                text_height = painter.fontMetrics().height()
                
                text_x = self.cursor_pos.x() - text_width // 2
                text_y = self.cursor_pos.y() - text_height - 5
                
                # 텍스트 배경 그리기
                painter.setBrush(QColor(0, 0, 0, 150))  # 반투명 검은색 배경
                painter.drawRect(text_x - 3, text_y, text_width + 5, text_height + 5)
                
                # 텍스트 그리기
                painter.drawText(text_x, text_y + text_height, text)

    def update_lines_and_midpoints(self):
        """점들을 기반으로 선과 중간점을 업데이트하는 메서드"""
        self.lines = []
        self.midpoints = []
        for i in range(0, len(self.points), 2):
            if i + 1 < len(self.points):
                # 두 점 사이의 중간점 계산
                midpoint = QPoint((self.points[i].x() + self.points[i+1].x()) // 2,
                                  (self.points[i].y() + self.points[i+1].y()) // 2)
                self.midpoints.append(midpoint)

        # 중간점들을 연결하여 선 생성
        for i in range(0, len(self.midpoints), 2):
            if i + 1 < len(self.midpoints):
                self.lines.append((self.midpoints[i], self.midpoints[i+1]))

    def save_labels(self):
        """라벨 데이터를 JSON 형식으로 저장하는 메서드"""
        if not self.labeling_dir:
            self.change_save_directory()
            if not self.labeling_dir:
                self.info_label.setText('Labeling directory not set. Labels not saved.')
                return

        if self.image_paths:
            label_data = {
                'image': os.path.basename(self.image_paths[self.current_image_index]),
                'lines': []
            }

            if self.lines:  # 라벨링된 선이 있는 경우
                for i, line in enumerate(self.lines):
                    start, end = line
                    cx = (start.x() + end.x()) / 2
                    cy = (start.y() + end.y()) / 2
                    dx = end.x() - start.x()
                    dy = end.y() - start.y()
                    angle = math.atan2(dy, dx)
                    length = math.sqrt(dx**2 + dy**2)
                    
                    vertices = self.points[i*4:(i*4)+4]
                    vertices_coords = [[p.x(), p.y()] for p in vertices]

                    label_data['lines'].append({
                        'center': [cx, cy],
                        'angle': angle,
                        'length': length,
                        'vertices': vertices_coords
                    })
            
            os.makedirs(self.labeling_dir, exist_ok=True)
            json_filename = os.path.join(self.labeling_dir, os.path.splitext(os.path.basename(self.image_paths[self.current_image_index]))[0] + '.json')
            
            with open(json_filename, 'w') as f:
                json.dump(label_data, f, indent=2)  # 라벨 데이터를 JSON 파일로 저장
            
            self.info_label.setText(f'Labels saved to {json_filename}')
            QMessageBox.information(self, "Save Successful", f"Labels saved to {json_filename}")
            
            self.labeled_images.add(self.image_paths[self.current_image_index])
            self.update_labeled_status()
            self.update_progress_label()  # 진행 상황 업데이트
        else:
            self.info_label.setText('No image loaded. Labels not saved.')
            QMessageBox.warning(self, "Save Failed", "No image loaded.")

    def check_if_labeled(self, image_path):
        """이미지가 이미 라벨링되었는지 확인하는 메서드"""
        if not self.labeling_dir:
            return False
        json_filename = os.path.join(self.labeling_dir, os.path.splitext(os.path.basename(image_path))[0] + '.json')
        return os.path.exists(json_filename)  # 해당 이미지의 JSON 파일이 존재하는지 확인

    def update_labeled_status(self):
        """라벨링 상태를 업데이트하고 UI에 반영하는 메서드"""
        current_item = self.image_list.item(self.current_image_index)
        current_path = self.image_paths[self.current_image_index]
        if current_path in self.labeled_images:
            current_item.setText(f"[완료] {os.path.basename(current_path)}")
        else:
            current_item.setText(os.path.basename(current_path))

    def clear_all(self):
        """모든 점과 선을 지우는 메서드"""
        self.points = []
        self.midpoints = []
        self.lines = []
        self.temp_line = None
        self.update()
        self.info_label.setText('All cleared. Start adding new points.')

    def undo_last_point(self):
        """마지막으로 찍은 점을 취소하는 메서드"""
        if self.points:
            self.points.pop()  # 마지막 점 제거
            if len(self.points) % 2 == 0 and self.midpoints:
                self.midpoints.pop()  # 마지막 중간점 제거
            if self.lines:
                self.lines.pop()  # 마지막 선 제거
            self.update()
            self.info_label.setText(f'Undo successful. Points: {len(self.points)}, Midpoints: {len(self.midpoints)}')

    def update_progress_label(self):
        """라벨링 진행 상황을 업데이트하는 메서드"""
        labeled_count = len(self.labeled_images)
        total_count = len(self.image_paths)
        self.progress_label.setText(f'Labeled: {labeled_count} / {total_count}')

    def keyPressEvent(self, event):
        """키 입력 이벤트를 처리하는 메서드"""
        super().keyPressEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)  # QApplication 인스턴스 생성
    ex = LabelingTool()  # LabelingTool 인스턴스 생성
    ex.show()  # 애플리케이션 윈도우 표시
    sys.exit(app.exec_())  # 애플리케이션의 이벤트 루프 시작