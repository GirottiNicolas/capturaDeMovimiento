import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración de MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,      # Modo dinámico para video
    max_num_hands=2,             # Detectar hasta 2 manos
    min_detection_confidence=0.5, # Confianza mínima para detectar
    min_tracking_confidence=0.5   # Confianza mínima para seguimiento
)

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)

print("Presiona 'q' para salir")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Convertir la imagen a RGB (MediaPipe requiere este formato)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar manos
    results = hands.process(rgb_frame)

    # Si se detectan manos, dibujar las articulaciones
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar los puntos y conexiones de las manos
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Configuración de puntos
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # Configuración de conexiones
            )

    # Mostrar el video con las manos detectadas
    cv2.imshow("Detección de articulaciones de las manos", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
hands.close()
