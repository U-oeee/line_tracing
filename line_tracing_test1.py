import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines, color=(0, 255, 0), thickness=5):
    if lines is None:
        return
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def average_slope_intercept(lines):
    left_fit  = []
    right_fit = []
    if lines is None:
        return None, None
    for x1, y1, x2, y2 in lines:
        # avoid divide by zero
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if slope < -0.5:  # 왼쪽 차선
            left_fit.append((slope, intercept))
        elif slope > 0.5: # 오른쪽 차선
            right_fit.append((slope, intercept))
    left_line  = np.mean(left_fit,  axis=0) if left_fit  else None
    right_line = np.mean(right_fit, axis=0) if right_fit else None
    return left_line, right_line

def make_line_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    # x = (y - b) / m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]

def compute_direction(vp_x, img_center_x, thresh=50):
    offset = vp_x - img_center_x
    if abs(offset) < thresh:
        return "Straight"
    return "Left" if offset < 0 else "Right"

def main():
    cap = cv2.VideoCapture(0)  # 또는 '/dev/video0'
    if not cap.isOpened():
        print("Camera open failed")
        return

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    roi_vertices = np.array([[
        (0, h),
        (w, h),
        (w, int(h * 0.6)),
        (0, int(h * 0.6))
    ]], dtype=np.int32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) 그레이스케일 & 2) 이진화(threshold)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 3) 블러 & 4) 엣지 검출
        blur = cv2.GaussianBlur(binary, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 5) ROI 마스킹
        masked = region_of_interest(edges, roi_vertices)

        # 6) 허프 변환으로 직선 검출
        lines = cv2.HoughLinesP(masked, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=150)

        # 7) 좌·우 차선 평균 모델 생성
        left_line, right_line = average_slope_intercept(lines)

        # 8) 프레임 하단(y1)에서 ROI 상단(y2)까지 직선 좌표 생성
        y1 = h
        y2 = int(h * 0.6)
        left_pts  = make_line_points(y1, y2, left_line)
        right_pts = make_line_points(y1, y2, right_line)

        # 9) 차선 그리기
        line_img = np.zeros_like(frame)
        if left_pts is not None:
            draw_lines(line_img, [left_pts])
        if right_pts is not None:
            draw_lines(line_img, [right_pts])
        combo = cv2.addWeighted(frame, 0.8, line_img, 1, 0)

        # 10) 소실점 계산 (두 직선의 교점)
        if left_line is not None and right_line is not None:
            m1, b1 = left_line
            m2, b2 = right_line
            vp_x = int((b2 - b1) / (m1 - m2))
            vp_y = int(m1 * vp_x + b1)
            cv2.circle(combo, (vp_x, vp_y), 8, (0, 0, 255), -1)
        else:
            vp_x = w // 2  # fallback

        # 11) 방향 결정 및 표시
        direction = compute_direction(vp_x, w // 2)
        cv2.putText(combo, direction, (int(w*0.4), 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        cv2.imshow("Lane Detection", combo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
