from flask import Flask, render_template, session
from image_processing import image_app
from video_processing import video_app
import os

# Creación de la aplicación Flask
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Establecimiento de una clave secreta para la sesión de Flask

# Directorio para almacenar las imágenes procesadas
PROCESSED_IMAGES_DIR = 'processed_images'
app.config['PROCESSED_IMAGES_DIR'] = PROCESSED_IMAGES_DIR

# Directorio para almacenar los videos procesados
PROCESSED_VIDEOS_DIR = 'processed_videos'
app.config['PROCESSED_VIDEOS_DIR'] = PROCESSED_VIDEOS_DIR

# Función para verificar y crear los directorios necesarios
def check_and_create_directories():
    if not os.path.exists(PROCESSED_IMAGES_DIR):
        os.makedirs(PROCESSED_IMAGES_DIR)
    if not os.path.exists(PROCESSED_VIDEOS_DIR):
        os.makedirs(PROCESSED_VIDEOS_DIR)

# Ruta para la página principal
@app.route('/')
def index():
    # Renderiza la plantilla de la página principal
    return render_template('index.html')

# Registra los Blueprints para el procesamiento de imágenes y videos en la aplicación Flask
app.register_blueprint(image_app)
app.register_blueprint(video_app)

if __name__ == '__main__':
    # Verifica y crea los directorios necesarios antes de iniciar la aplicación Flask
    check_and_create_directories()
    app.run(host='0.0.0.0', debug=True)
