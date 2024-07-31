import cv2
import mediapipe as mp


def detect_hand_landmarks(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            connections = mp_hands.HAND_CONNECTIONS
            for connection in connections:
                idx1 = connection[0]
                idx2 = connection[1]
                x1, y1 = int(hand_landmarks.landmark[idx1].x * w), int(hand_landmarks.landmark[idx1].y * h)
                x2, y2 = int(hand_landmarks.landmark[idx2].x * w), int(hand_landmarks.landmark[idx2].y * h)
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return image


def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_hand_landmarks(frame)

        cv2.imshow('Hand Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
