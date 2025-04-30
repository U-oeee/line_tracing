import cv2
import numpy as np

cap = cv2.VideoCapture('/dev/video0')
width, height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def draw_arrow(frame, direction):
    center = (width // 2, 50)
    if direction == "Left":
        cv2.arrowedLine(frame, center, (center[0] - 100, center[1]), (0, 0, 255), 5, tipLength=0.5)
    elif direction == "Right":
        cv2.arrowedLine(frame, center, (center[0] + 100, center[1]), (0, 0, 255), 5, tipLength=0.5)
    else:
        cv2.arrowedLine(frame, (center[0], center[1]+30), (center[0], center[1]-30), (0, 255, 0), 5, tipLength=0.5)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(binary, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ROI 영역 재설정 (작고 정중앙 위주)
    mask = np.zeros_like(edges)
    vertices = np.array([[
        (width * 0.3, height * 0.8),
        (width * 0.4, height * 0.5),
        (width * 0.6, height * 0.5),
        (width * 0.7, height * 0.8)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    roi = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 30, minLineLength=30, maxLineGap=20)

    left_lines, right_lines = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.5:
                left_lines.append(line[0])
            elif slope > 0.5:
                right_lines.append(line[0])

    def fit_line(lines):
        if not lines:
            return None
        points = np.vstack([[x1, y1] for x1, y1, x2, y2 in lines] +
                           [[x2, y2] for x1, y1, x2, y2 in lines])
        [vx, vy, x, y] = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        slope = vy / vx
        intercept = y - slope * x
        return slope[0], intercept[0]

    y1, y2 = height, int(height * 0.6)
    lane_center = width // 2

    left_x1 = left_x2 = right_x1 = right_x2 = None

    if (l := fit_line(left_lines)) is not None:
        m, b = l
        left_x1 = int((y1 - b) / m)
        left_x2 = int((y2 - b) / m)
        cv2.line(frame, (left_x1, y1), (left_x2, y2), (0, 255, 0), 4)

    if (r := fit_line(right_lines)) is not None:
        m, b = r
        right_x1 = int((y1 - b) / m)
        right_x2 = int((y2 - b) / m)
        cv2.line(frame, (right_x1, y1), (right_x2, y2), (0, 255, 0), 4)

    direction = "Straight"

    if left_x1 and right_x1:
        lane_center = (left_x1 + right_x1) // 2
        offset = lane_center - (width // 2)
        if offset < -30:
            direction = "Left"
        elif offset > 30:
            direction = "Right"

    draw_arrow(frame, direction)
    cv2.putText(frame, f"{direction}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    output = process_frame(frame)
    cv2.imshow("Lane Tracker", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
