import cv2

# Cargar la imagen y convertirla a escala de grises
img = cv2.imread('NGC 6397.tif', cv2.IMREAD_GRAYSCALE)

Alto, ancho, = img.shape[:2]

escala = 0.22

img_resized = cv2.resize(img, (int(ancho * escala), int(Alto * escala)))

# Aplicar un filtro gaussiano para reducir el ruido
img_blur = cv2.GaussianBlur(img, (5, 5), 1.5)

# Umbralizar la imagen para obtener una imagen binaria
_, img_thresh = cv2.threshold(img_blur, 130, 255, cv2.THRESH_BINARY)

# Encontrar los contornos en la imagen binaria
contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos sobre una imagen en blanco
img_contours = cv2.drawContours(img.copy(), contours, -1, (150, 30, 215), -1)

# Contar y mostrar el número de estrellas
num_stars = len(contours)
print(f"Number of stars detected: {num_stars}")

# Mostrar la imagen original y la imagen con contornos
cv2.imshow('Resized Image', img)
cv2.imshow('Stars Counted', img_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
    