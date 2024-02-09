import cv2

# Étape 1: Charger l'image
image = cv2.imread('image test 1.png')

# Étape 2: Définir les coordonnées de la région d'intérêt (ROI)
# Supposons que vous souhaitiez extraire un rectangle situé à (x_start=100, y_start=50) et de taille 200x150 pixels
x_start, y_start, width, height = 456, 247, 124, 209 // 2
x_end, y_end = x_start + width, y_start + height

# Étape 3: Extraire la ROI de l'image
roi = image[y_start:y_end, x_start:x_end]

# Étape 4: Afficher la ROI
cv2.imshow('Region d interet', roi)

# Attendre que l'utilisateur appuie sur une touche pour fermer la fenêtre
cv2.waitKey(0)
cv2.destroyAllWindows()
