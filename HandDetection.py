import cv2
import mediapipe as mp

# Inicializa a MediaPipe para mãos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Inicializa o modelo de detecção de mãos
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem")
            break

        # Converte a imagem para RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processa a imagem e detecta mãos
        result = hands.process(image_rgb)

        # Desenha as landmarks das mãos na imagem original
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Mostra a imagem com a detecção
        cv2.imshow('Detecção de Mãos', frame)

        # Pressione 'q' para sair do loop
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()
