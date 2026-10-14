import os
import io
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from torchvision import transforms, models
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

app = Flask(__name__)
app.secret_key = 'agroai_secret_key_fitosanitario_palmas'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agroai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(120), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    diagnosis = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)

with app.app_context():
    db.create_all()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ['MANCHAS_FOLIARES', 'MARCHITEZ_AVANZADA', 'MARCHITEZ_SORPRESIVA', 'SANA']
NUM_CLASSES = len(CLASSES)

MODELO_SEG_PATH = os.path.join('weights', 'best.pt')
MODEL_DETECTOR_PATH = os.path.join('weights', 'detector_palma_vs_nopalma.pkl')
MODEL_EFF_PATH = os.path.join('weights', 'best_effb0_palma_model.pth')

model_seg = None
detector_dominio = None
model_eff = None

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_all_models():
    global model_seg, detector_dominio, model_eff
    
    # 1. YOLO Segmentación (Bokeh)
    if os.path.exists(MODELO_SEG_PATH) and YOLO is not None:
        try:
            model_seg = YOLO(MODELO_SEG_PATH)
            print(f"[*] Modelo de Segmentación YOLO cargado desde {MODELO_SEG_PATH}")
        except Exception as e:
            print(f"[!] Error al cargar YOLO: {e}")

    # 2. Detector Anti-OOD (Joblib)
    if os.path.exists(MODEL_DETECTOR_PATH):
        try:
            detector_dominio = joblib.load(MODEL_DETECTOR_PATH)
            print(f"[*] Detector Anatómico Anti-OOD cargado desde {MODEL_DETECTOR_PATH}")
        except Exception as e:
            print(f"[!] Error al cargar detector: {e}")

    # 3. Clasificador EfficientNet-B0
    if os.path.exists(MODEL_EFF_PATH):
        try:
            model_eff = models.efficientnet_b0(weights=None)
            in_features = model_eff.classifier[1].in_features
            model_eff.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, NUM_CLASSES)
            )
            model_eff.load_state_dict(torch.load(MODEL_EFF_PATH, map_location=DEVICE))
            model_eff.to(DEVICE)
            model_eff.eval()
            print(f"[*] Clasificador EfficientNet-B0 cargado desde {MODEL_EFF_PATH}")
        except Exception as e:
            print(f"[!] Error al cargar EfficientNet-B0: {e}")

load_all_models()

RECOMMENDATIONS = {
    'MANCHAS_FOLIARES': {
        'severidad': 'Afectación Moderada',
        'color_badge': 'bg-warning text-dark',
        'acciones': [
            'Realizar poda sanitaria de las hojas con síntomas avanzados para reducir inóculo.',
            'Aplicar fungicida protectante/sistémico a base de triazoles o estrobilurinas.',
            'Verificar y mejorar el sistema de drenaje para prevenir acumulación de humedad.',
            'Monitorear la parcela cada 8 días para evitar diseminación.'
        ]
    },
    'MARCHITEZ_AVANZADA': {
        'severidad': 'Severidad Alta / Crítica',
        'color_badge': 'bg-danger text-white',
        'acciones': [
            'Aislar inmediatamente la palma para evitar transmisión radicular o por vectores.',
            'Evaluar erradicación de la palma afectada siguiendo protocolos del ICA.',
            'Desinfectar herramientas de poda con hipoclorito de sodio al 10% entre plantas.',
            'Tomar muestras de raíz/suelo para confirmación fitopatológica en laboratorio.'
        ]
    },
    'MARCHITEZ_SORPRESIVA': {
        'severidad': 'Severidad Alta / Cuarentenaria',
        'color_badge': 'bg-danger text-white',
        'acciones': [
            'Inspección focalizada del vector chinche de la palma (Lincus sp.).',
            'Aplicar control integrado de vectores en el lote y lotes vecinos.',
            'Erradicación inmediata de la palma afectada si hay marchitamiento acelerado.',
            'Reportar el foco al departamento técnico agronómico de la zona.'
        ]
    },
    'SANA': {
        'severidad': 'Estado Óptimo / Sin Síntomas',
        'color_badge': 'bg-success text-white',
        'acciones': [
            'Mantener el plan nutricional y programa de fertilización de rutina.',
            'Continuar con los plateos y limpiezas programadas del cultivo.',
            'Realizar monitoreo preventivo quincenal en hojas del estrato medio.'
        ]
    }
}

def aplicar_bokeh_yolo(pil_img):
    if model_seg is None:
        return pil_img
    
    img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = img_np.shape
    
    try:
        results = model_seg(img_np, verbose=False)[0]
        mask = np.zeros((h, w), dtype=np.float32)
        if results.masks is not None:
            for m in results.masks.data:
                m_np = m.cpu().numpy().astype(np.float32)
                m_resized = cv2.resize(m_np, (w, h))
                mask = np.maximum(mask, m_resized)

        fondo_borroso = cv2.GaussianBlur(img_np, (99, 99), 0)
        mask_soft = cv2.GaussianBlur(mask, (11, 11), 0)
        mask_3ch = np.repeat(mask_soft[:, :, np.newaxis], 3, axis=2)
        
        resultado = (img_np * mask_3ch + fondo_borroso * (1.0 - mask_3ch)).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"[!] Error en Bokeh YOLO: {e}")
        return pil_img

def extraer_descriptores_fisionomicos(img_pil):
    img_np = np.array(img_pil.resize((128, 128)))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    edges = cv2.Canny(gray_eq, 30, 130)
    densidad_bordes = np.sum(edges > 0) / (128.0 * 128.0)

    gx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist_angle, _ = np.histogram(angle, bins=12, range=(0, 360), weights=mag)
    hist_angle = hist_angle / (np.sum(hist_angle) + 1e-6)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mean_hsv = np.mean(hsv, axis=(0, 1))
    std_hsv = np.std(hsv, axis=(0, 1))

    return np.hstack([densidad_bordes, hist_angle, mean_hsv, std_hsv])

def generate_confidence_chart(probabilities, classes, predicted_class, confidence_val):
    display_classes = [c.replace('_', ' ') for c in classes]
    fig, ax = plt.subplots(figsize=(5, 4))
    y_pos = np.arange(len(display_classes))
    probs = [p * 100 for p in probabilities]
    
    ax.barh(y_pos, probs, color='#10b981', height=0.45)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_classes, fontsize=9, fontweight='bold', color='#333333')
    ax.invert_yaxis()
    ax.set_xlabel('Confianza (%)', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_title(f"Predicción: {predicted_class} ({confidence_val:.1f}%)", fontsize=10, fontweight='bold')
    
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def pil_to_base64(pil_img):
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(pil_img)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        session['user'] = user.email
        session['user_name'] = user.name
        return jsonify({'success': True, 'user_name': user.name})
    return jsonify({'success': False, 'error': 'Correo o contraseña incorrectos'}), 401

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    if not name or not email or not password:
        return jsonify({'success': False, 'error': 'Todos los campos son obligatorios'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'error': 'El correo ya se encuentra registrado'}), 400
    
    new_user = User(name=name, email=email)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    
    session['user'] = new_user.email
    session['user_name'] = new_user.name
    return jsonify({'success': True, 'user_name': new_user.name})

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

@app.route('/')
def index():
    user_name = session.get('user_name', None)
    return render_template('index.html', user_name=user_name)

@app.route('/history', methods=['GET'])
def get_history():
    if 'user' not in session:
        return jsonify({'records': []})
    user_email = session['user']
    records = History.query.filter_by(user_email=user_email).order_by(History.id.desc()).limit(15).all()
    out = []
    for r in records:
        out.append({
            'timestamp': r.timestamp,
            'filename': r.filename,
            'diagnosis': r.diagnosis,
            'confidence': r.confidence
        })
    return jsonify({'records': out})

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return jsonify({'error': 'Acceso denegado. Inicia sesión.'}), 401
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Archivo no seleccionado'}), 400
        
    try:
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # ETAPA 1: Segmentación Bokeh con YOLO (best.pt)
        bokeh_img = aplicar_bokeh_yolo(pil_img)
        
        # ETAPA 2: Detector Anti-OOD (Palma vs No Palma)
        if detector_dominio is not None:
            feat = extraer_descriptores_fisionomicos(bokeh_img).reshape(1, -1)
            es_palma_valida = detector_dominio.predict(feat)[0]
            if es_palma_valida == -1:
                return jsonify({
                    'success': False,
                    'error': 'La imagen cargada no corresponde a una hoja de palma de aceite válida o presenta anomalías estructurales.'
                }), 400

        # ETAPA 3: Clasificación con EfficientNet-B0 (best_effb0_palma_model.pth)
        if model_eff is not None:
            tensor_img = val_transforms(bokeh_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model_eff(tensor_img)
                probs = F.softmax(outputs, dim=1)
                probabilities = probs.cpu().numpy()[0].tolist()
        else:
            probabilities = [0.25, 0.25, 0.25, 0.25]
            
        pred_idx = int(np.argmax(probabilities))
        raw_class = CLASSES[pred_idx]
        predicted_class = raw_class.replace('_', ' ')
        confidence = probabilities[pred_idx] * 100
        
        chart_b64 = generate_confidence_chart(probabilities, CLASSES, predicted_class, confidence)
        img_entrada_b64 = pil_to_base64(bokeh_img)
        
        # Grad-CAM / Visualización de segmentación YOLO
        if model_seg is not None:
            res_yolo = model_seg(pil_img, verbose=False)[0]
            gradcam_bgr = res_yolo.plot()
            gradcam_rgb = cv2.cvtColor(gradcam_bgr, cv2.COLOR_BGR2RGB)
            gradcam_b64 = pil_to_base64(gradcam_rgb)
        else:
            gradcam_b64 = pil_to_base64(bokeh_img)
        
        recom_data = RECOMMENDATIONS.get(raw_class, RECOMMENDATIONS['SANA'])
        
        # Guardar en Historial SQLite
        from datetime import datetime
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_hist = History(
            user_email=session['user'],
            filename=file.filename,
            diagnosis=predicted_class,
            confidence=f"{confidence:.1f}%",
            timestamp=now_str
        )
        db.session.add(new_hist)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.1f}%",
            'severidad': recom_data['severidad'],
            'color_badge': recom_data['color_badge'],
            'acciones': recom_data['acciones'],
            'img_entrada': f"data:image/png;base64,{img_entrada_b64}",
            'gradcam': f"data:image/png;base64,{gradcam_b64}",
            'chart': f"data:image/png;base64,{chart_b64}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
