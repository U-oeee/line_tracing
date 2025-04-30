import cv2
import numpy as np

# 카메라 설정
cap = cv2.VideoCapture('/dev/video0')
width, height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def process_frame(frame):
    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 이진화 처리 (흰색 차선 강조)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 가우시안 블러 적용
    blur = cv2.GaussianBlur(binary, (5,5), 0)
    
    # Canny 엣지 검출
    edges = cv2.Canny(blur, 50, 150)
    
    # ROI(관심 영역) 설정
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (width*0.1, height*0.9),
        (width*0.4, height*0.6),
        (width*0.6, height*0.6),
        (width*0.9, height*0.9)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 허프 변환을 이용한 직선 검출
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50,
                           minLineLength=50, maxLineGap=30)
    
    # 차선 정보 초기화
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2-y1)/(x2-x1) if (x2-x1) != 0 else 0
            if slope < -0.5:  # 왼쪽 차선
                left_lines.append(line[0])
            elif slope > 0.5:  # 오른쪽 차선
                right_lines.append(line[0])
    
    # 차선 평균화 처리
    def average_lines(lines):
        if not lines:
            return None
        xs, ys = [], []
        for line in lines:
            xs.extend([line[0], line[2]])
            ys.extend([line[1], line[3]])
        poly = np.polyfit(ys, xs, 1)
        return poly
    
    left_poly = average_lines(left_lines)
    right_poly = average_lines(right_lines)
    
    # 차선 시각화 및 방향 계산
    lane_center = width//2
    if left_poly is not None and right_poly is not None:
        # 차선 위치 계산
        y1 = height
        y2 = int(height*0.6)
        
        left_x1 = int(np.polyval(left_poly, y1))
        left_x2 = int(np.polyval(left_poly, y2))
        right_x1 = int(np.polyval(right_poly, y1))
        right_x2 = int(np.polyval(right_poly, y2))
        
        # 차선 중앙 계산
        lane_center = (left_x1 + right_x1) // 2
        
        # 차선 그리기
        cv2.line(frame, (left_x1, y1), (left_x2, y2), (0,255,0), 5)
        cv2.line(frame, (right_x1, y1), (right_x2, y2), (0,255,0), 5)
    
    # 방향 표시
    direction = "Straight"
    offset = lane_center - width//2
    if offset < -50:
        direction = "Left"
    elif offset > 50:
        direction = "Right"
    
    cv2.putText(frame, f"Direction: {direction}", (20,40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed = process_frame(frame)
    cv2.imshow('Lane Detection', processed)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
