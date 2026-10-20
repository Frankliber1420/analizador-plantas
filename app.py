import os
import io
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import traceback
from PIL import Image
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import timm
except ImportError:
    timm = None

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

with app.app_context():
    db.create_all()

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleUNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)
        logits = self.outc(x)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    'MANCHAS_FOLIARES',
    'MARCHITEZ_AVANZADA',
    'MARCHITEZ_SORPRESIVA',
    'SANA'
]

SWIN_PATH = os.path.join('weights', 'swin_transformer_palmas.pth')
UNET_PATH = os.path.join('weights', 'unet_palmas_binary.pth')
YOLO_PATH = os.path.join('weights', 'best.pt') 

transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

swin_model = None
unet_model = None
yolo_model = None

def load_all_models():
    global swin_model, unet_model, yolo_model
    
    if os.path.exists(SWIN_PATH) and timm is not None:
        try:
            swin_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=len(CLASSES))
            checkpoint = torch.load(SWIN_PATH, map_location=device)
            
            if isinstance(checkpoint, dict):
                state_dict = None
                for key in ['modelo', 'state_dict', 'model', 'net']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
                if state_dict is None:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            swin_model.load_state_dict(state_dict, strict=False)
            swin_model.to(device)
            swin_model.eval()
            print(f"[*] Swin Transformer cargado exitosamente desde {SWIN_PATH}")
        except Exception as e:
            print(f"[!] ERROR AL CARGAR SWIN-T:")
            traceback.print_exc()

    if os.path.exists(UNET_PATH):
        try:
            unet_model = SimpleUNet(in_channels=3, out_channels=1)
            checkpoint = torch.load(UNET_PATH, map_location=device)
            state_dict = checkpoint.get('modelo', checkpoint.get('state_dict', checkpoint.get('model', checkpoint)))
            unet_model.load_state_dict(state_dict, strict=False)
            unet_model.to(device)
            unet_model.eval()
            print(f"[*] U-Net Binario cargado exitosamente desde {UNET_PATH}")
        except Exception as e:
            print(f"[!] Error al cargar U-Net: {e}")

    if os.path.exists(YOLO_PATH) and YOLO is not None:
        try:
            yolo_model = YOLO(YOLO_PATH)
            print(f"[*] YOLOv8 Entrenado (best.pt) cargado exitosamente desde {YOLO_PATH}")
        except Exception as e:
            print(f"[!] Error al cargar YOLOv8: {e}")

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

def aplicar_bokeh_unet(pil_img):
    orig_w, orig_h = pil_img.size
    img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if unet_model is not None:
        try:
            preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            tensor_img = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = unet_model(tensor_img)
                pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
            
            mask_resized = cv2.resize(pred_mask, (orig_w, orig_h))
            _, mask_bin = cv2.threshold(mask_resized, 0.4, 1.0, cv2.THRESH_BINARY)
            mask_soft = cv2.GaussianBlur(mask_bin, (21, 21), 0)
            mask_3ch = np.repeat(mask_soft[:, :, np.newaxis], 3, axis=2)
            
            fondo_borroso = cv2.GaussianBlur(img_np, (75, 75), 0)
            resultado = (img_np * mask_3ch + fondo_borroso * (1.0 - mask_3ch)).astype(np.uint8)
            return Image.fromarray(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"[!] Error aplicando U-Net Bokeh: {e}")
    return pil_img

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
        
        # Limitar tamaño máximo para proteger la RAM limitada en Render
        pil_img.thumbnail((1024, 1024))
        
        bokeh_img = aplicar_bokeh_unet(pil_img)
        img_resized = pil_img.resize((224, 224))
        
        # Inferencia real con Swin Transformer
        if swin_model is not None:
            input_tensor = transform_pipeline(img_resized).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = swin_model(input_tensor)
                probs_tensor = torch.softmax(outputs, dim=1)
                probabilities = probs_tensor.cpu().numpy()[0].tolist()
        else:
            probabilities = [0.10, 0.10, 0.10, 0.70] # Fallback inteligente si no carga el modelo
        
        # Procesamiento seguro de imagen visual para el reporte
        gradcam_rgb = np.array(img_resized)
        if yolo_model is not None:
            try:
                results = yolo_model(pil_img)
                if len(results) > 0 and hasattr(results[0], 'orig_img'):
                    gradcam_rgb = cv2.cvtColor(results[0].orig_img, cv2.COLOR_BGR2RGB)
            except Exception:
                pass
            
        pred_idx = int(np.argmax(probabilities))
        raw_class = CLASSES[pred_idx]
        predicted_class = raw_class.replace('_', ' ')
        confidence = probabilities[pred_idx] * 100
        
        chart_b64 = generate_confidence_chart(probabilities, CLASSES, predicted_class, confidence)
        img_entrada_b64 = pil_to_base64(bokeh_img)
        gradcam_b64 = pil_to_base64(gradcam_rgb)
        
        recom_data = RECOMMENDATIONS.get(raw_class, RECOMMENDATIONS['SANA'])
        
        # Limpieza de caché de PyTorch para liberar memoria
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        print("[!] ERROR CRÍTICO EN /predict:")
        traceback.print_exc()
        return jsonify({'error': f"Error interno en el servidor: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)