import os
import cv2
import pandas as pd
import mediapipe as mp
from datetime import datetime
from flask import Flask, Blueprint, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)

video_app = Blueprint('video_app', __name__)

app.config.update(
    UPLOAD_FOLDER='uploads',
    PROCESSED_FOLDER='static/processed',
    ALLOWED_EXTENSIONS={'mp4', 'avi', 'mov'},
    FPS=20  # Tasa de fotogramas por segundo para el análisis
)

# Asegurarse de que la carpeta de procesados exista
if not os.path.exists(app.config['PROCESSED_FOLDER']):
    os.makedirs(app.config['PROCESSED_FOLDER'])

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def calcular_angulo(p1, p2, p3):

    """Calcula el ángulo entre tres puntos usando la fórmula de coseno."""
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])
    
    ab = a - b
    bc = c - b
    cos_theta = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

def determinar_codigo_postura_espalda(pose_landmarks):
    """Determina el código de postura para la espalda basado en los puntos de referencia."""
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    flexion_angle = calcular_angulo(left_shoulder, left_hip, right_hip)
    lateral_angle = calcular_angulo(left_shoulder, right_shoulder, right_hip)
    rotation_angle = calcular_angulo(left_shoulder, left_hip, right_shoulder)

    if flexion_angle < 20 and lateral_angle < 20 and rotation_angle < 20:
        return 1
    elif flexion_angle >= 20 and lateral_angle < 20 and rotation_angle < 20:
        return 2
    elif (flexion_angle < 20 and lateral_angle >= 20) or (rotation_angle >= 20):
        return 3
    elif flexion_angle >= 20 and (lateral_angle >= 20 or rotation_angle >= 20):
        return 4

def determinar_codigo_postura_brazos(pose_landmarks):
    """Determina el código de postura para los brazos basado en los puntos de referencia."""
    left_elbow = pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    left_arm_above_shoulder = left_elbow.y < left_shoulder.y
    right_arm_above_shoulder = right_elbow.y < right_shoulder.y

    if not left_arm_above_shoulder and not right_arm_above_shoulder:
        return 1
    elif left_arm_above_shoulder != right_arm_above_shoulder:
        return 2
    elif left_arm_above_shoulder and right_arm_above_shoulder:
        return 3

def determinar_codigo_postura_piernas(pose_landmarks):
    """Determina el código de postura para las piernas basado en los puntos de referencia."""
    return 1

def determinar_codigo_postura_carga(peso_carga):
    """Determina el código de postura para la carga basado en el peso."""
    return 1  # La carga es siempre 1

def obtener_nivel_riesgo(codigo_espalda, codigo_brazos, codigo_piernas, codigo_carga):
    
    
    riesgo_tabla = {
        (1, 1, 1, 1): 1, (1, 1, 1, 2): 1, (1, 1, 1, 3): 1, (1, 1, 2, 1): 1, (1, 1, 2, 2): 1, (1, 1, 2, 3): 1,
    (1, 1, 3, 1): 1, (1, 1, 3, 2): 1, (1, 1, 3, 3): 1, (1, 2, 1, 1): 1, (1, 2, 1, 2): 1, (1, 2, 1, 3): 1,
    (1, 2, 2, 1): 1, (1, 2, 2, 2): 1, (1, 2, 2, 3): 1, (1, 2, 3, 1): 1, (1, 2, 3, 2): 1, (1, 2, 3, 3): 1,
    (1, 3, 1, 1): 1, (1, 3, 1, 2): 1, (1, 3, 1, 3): 1, (1, 3, 2, 1): 1, (1, 3, 2, 2): 1, (1, 3, 2, 3): 1,
    (1, 3, 3, 1): 1, (1, 3, 3, 2): 1, (1, 3, 3, 3): 1, (2, 1, 1, 1): 2, (2, 1, 1, 2): 2, (2, 1, 1, 3): 3,
    (2, 1, 2, 1): 2, (2, 1, 2, 2): 2, (2, 1, 2, 3): 3, (2, 1, 3, 1): 2, (2, 1, 3, 2): 2, (2, 1, 3, 3): 3,
    (2, 2, 1, 1): 2, (2, 2, 1, 2): 2, (2, 2, 1, 3): 3, (2, 2, 2, 1): 2, (2, 2, 2, 2): 2, (2, 2, 2, 3): 3,
    (2, 2, 3, 1): 2, (2, 2, 3, 2): 2, (2, 2, 3, 3): 3, (2, 3, 1, 1): 3, (2, 3, 1, 2): 3, (2, 3, 1, 3): 4,
    (2, 3, 2, 1): 2, (2, 3, 2, 2): 2, (2, 3, 2, 3): 3, (2, 3, 3, 1): 3, (2, 3, 3, 2): 3, (2, 3, 3, 3): 3,
    (3, 1, 1, 1): 1, (3, 1, 1, 2): 1, (3, 1, 1, 3): 1, (3, 1, 2, 1): 1, (3, 1, 2, 2): 1, (3, 1, 2, 3): 1,
    (3, 1, 3, 1): 1, (3, 1, 3, 2): 1, (3, 1, 3, 3): 2, (3, 2, 1, 1): 2, (3, 2, 1, 2): 2, (3, 2, 1, 3): 3,
    (3, 2, 2, 1): 1, (3, 2, 2, 2): 1, (3, 2, 2, 3): 1, (3, 2, 3, 1): 1, (3, 2, 3, 2): 1, (3, 2, 3, 3): 2,
    (3, 3, 1, 1): 2, (3, 3, 1, 2): 2, (3, 3, 1, 3): 3, (3, 3, 2, 1): 1, (3, 3, 2, 2): 1, (3, 3, 2, 3): 1,
    (3, 3, 3, 1): 2, (3, 3, 3, 2): 3, (3, 3, 3, 3): 3, (4, 1, 1, 1): 2, (4, 1, 1, 2): 3, (4, 1, 1, 3): 3,
    (4, 1, 2, 1): 2, (4, 1, 2, 2): 2, (4, 1, 2, 3): 3, (4, 1, 3, 1): 2, (4, 1, 3, 2): 2, (4, 1, 3, 3): 3,
    (4, 2, 1, 1): 3, (4, 2, 1, 2): 3, (4, 2, 1, 3): 4, (4, 2, 2, 1): 2, (4, 2, 2, 2): 3, (4, 2, 2, 3): 4,
    (4, 2, 3, 1): 3, (4, 2, 3, 2): 3, (4, 2, 3, 3): 4, (4, 3, 1, 1): 4, (4, 3, 1, 2): 4, (4, 3, 1, 3): 4,
    (4, 3, 2, 1): 2, (4, 3, 2, 2): 3, (4, 3, 2, 3): 4, (4, 3, 3, 1): 3, (4, 3, 3, 2): 3, (4, 3, 3, 3): 4
        
    }
    return riesgo_tabla.get((codigo_espalda, codigo_brazos, codigo_piernas, codigo_carga), 'Desconocido')

def get_next_conductor_id():
    id_file_path = 'static/conductor_id.txt'
    
    if os.path.exists(id_file_path):
        with open(id_file_path, 'r') as file:
            last_id = file.read().strip()
            next_id = int(last_id) + 1
    else:
        next_id = 1

    with open(id_file_path, 'w') as file:
        file.write(str(next_id))

    return f"Conductor {next_id}"

def process_video(filepath):
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y %H-%M-%S")
    
    video_filename = f"{timestamp} - video Analizado.mp4"
    excel_filename = "Resultados.xlsx"
    
    processed_video_path = os.path.join(app.config['PROCESSED_FOLDER'], video_filename)
    excel_path = os.path.join(app.config['PROCESSED_FOLDER'], excel_filename)

    cap = cv2.VideoCapture(filepath)
    frame_data = []
    frame_count = 0

    conductor_id = get_next_conductor_id()

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(static_image_mode=False) as pose:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_video_path, fourcc, app.config['FPS'], (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frame_count += 1
            current_frame = frame_count

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                back_code = determinar_codigo_postura_espalda(landmarks)
                arms_code = determinar_codigo_postura_brazos(landmarks)
                legs_code = determinar_codigo_postura_piernas(landmarks)
                load_code = determinar_codigo_postura_carga(1)  # Aunque no se usa, se sigue calculando
                risk_level = obtener_nivel_riesgo(back_code, arms_code, legs_code, load_code)

                frame_data.append({
                    'Conductor ID': conductor_id,
                    'Frame': current_frame,
                    'Espalda': back_code,
                    'Brazos': arms_code,
                    'Piernas': legs_code,
                    'Carga': load_code,  # Aunque no se usa en el resumen, se guarda en el Excel de resultados
                    'Riesgo': risk_level
                })

                # Dibujar los puntos de referencia en el fotograma original
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Guardar el fotograma procesado
            out.write(frame)

        cap.release()
        out.release()

    # Leer los datos existentes en el archivo Excel (si existe)
    if os.path.exists(excel_path):
        existing_df = pd.read_excel(excel_path)
        combined_df = pd.concat([existing_df, pd.DataFrame(frame_data)], ignore_index=True)
    else:
        combined_df = pd.DataFrame(frame_data)

    # Guardar los datos combinados en el archivo Excel
    combined_df.to_excel(excel_path, index=False)

    # Calcular frecuencias y porcentajes por tipo de código
    resumen_espalda = combined_df[combined_df['Conductor ID'] == conductor_id]['Espalda'].value_counts().reset_index()
    resumen_espalda.columns = ['Código', 'Frecuencia']
    resumen_espalda['Porcentaje (%)'] = (resumen_espalda['Frecuencia'] / resumen_espalda['Frecuencia'].sum()) * 100
    resumen_espalda['Parte del Cuerpo'] = 'Espalda'
    resumen_espalda['ID'] = conductor_id

    resumen_brazos = combined_df[combined_df['Conductor ID'] == conductor_id]['Brazos'].value_counts().reset_index()
    resumen_brazos.columns = ['Código', 'Frecuencia']
    resumen_brazos['Porcentaje (%)'] = (resumen_brazos['Frecuencia'] / resumen_brazos['Frecuencia'].sum()) * 100
    resumen_brazos['Parte del Cuerpo'] = 'Brazos'
    resumen_brazos['ID'] = conductor_id

    resumen_piernas = combined_df[combined_df['Conductor ID'] == conductor_id]['Piernas'].value_counts().reset_index()
    resumen_piernas.columns = ['Código', 'Frecuencia']
    resumen_piernas['Porcentaje (%)'] = (resumen_piernas['Frecuencia'] / resumen_piernas['Frecuencia'].sum()) * 100
    resumen_piernas['Parte del Cuerpo'] = 'Piernas'
    resumen_piernas['ID'] = conductor_id

    # Combinar los resúmenes en una sola tabla
    resumen_combinado = pd.concat([resumen_espalda, resumen_brazos, resumen_piernas], ignore_index=True)

    # Reordenar las columnas
    resumen_combinado = resumen_combinado[['ID', 'Parte del Cuerpo', 'Código', 'Frecuencia', 'Porcentaje (%)']]
    
    # Agregar una fila en blanco para separar los resultados de diferentes conductores
    blank_row = pd.DataFrame([[''] * len(resumen_combinado.columns)], columns=resumen_combinado.columns)

    # Leer el archivo de resumen existente (si existe)
    resumen_excel_path = os.path.join(app.config['PROCESSED_FOLDER'], 'Resumen.xlsx')
    if os.path.exists(resumen_excel_path):
        existing_summary_df = pd.read_excel(resumen_excel_path)
        # Concatenar los datos existentes con los nuevos y añadir una fila en blanco antes de los nuevos datos
        combined_summary_df = pd.concat([existing_summary_df, blank_row, resumen_combinado], ignore_index=True)
    else:
        # Si el archivo no existe, simplemente añadir los datos nuevos
        combined_summary_df = resumen_combinado

    # Guardar el resumen combinado en el archivo Excel con una sola hoja
    combined_summary_df.to_excel(resumen_excel_path, sheet_name='Resumen', index=False)

    return video_filename, excel_filename, 'Resumen.xlsx'


@video_app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            video_filename, excel_filename, resumen_filename = process_video(file_path)
            return redirect(url_for('video_app.video_resultado', video_filename=video_filename))
    return render_template('upload_video.html')

@video_app.route('/video_resultado/<video_filename>')
def video_resultado(video_filename):
    excel_filename = "Resultados.xlsx"
    resumen_filename = "Resumen.xlsx"
    excel_path = os.path.join(app.config['PROCESSED_FOLDER'], excel_filename)
    resumen_path = os.path.join(app.config['PROCESSED_FOLDER'], resumen_filename)

    df = pd.read_excel(excel_path)
    # Obtener el último ID de conductor para filtrar los resultados del video cargado
    conductor_id = df['Conductor ID'].iloc[-1]
    excel_data = df[df['Conductor ID'] == conductor_id].to_dict(orient='records')

    resumen_df = pd.read_excel(resumen_path)
    resumen_data = resumen_df[resumen_df['ID'] == conductor_id].to_dict(orient='records')

    return render_template('video_resultado.html', video_filename=video_filename, excel_data=excel_data, resumen_data=resumen_data)

@video_app.route('/download_results')
def download_results():
    excel_filename = "Resultados.xlsx"
    resumen_filename = "Resumen.xlsx"
    return redirect(url_for('static', filename='processed/' + excel_filename))

@video_app.route('/download_summary')
def download_summary():
    resumen_filename = "Resumen.xlsx"
    return redirect(url_for('static', filename='processed/' + resumen_filename))


app.register_blueprint(video_app)

if __name__ == '__main__':
    app.run(debug=True)