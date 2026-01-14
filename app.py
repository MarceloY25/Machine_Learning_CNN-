"""
Application de Détection de Maladies des Feuilles de Café
=========================================================
Cette application utilise un modèle de Deep Learning hybride pour analyser
des images de feuilles de café et déterminer leur état de santé.

Auteur: Groupe 8
Projet: Deep Learning - Classification des maladies des feuilles de café
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import io
import os

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Détection de Maladies - Feuilles de Café",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour une interface ultra-moderne et premium
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Variables CSS pour cohérence */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        --danger-gradient: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        --card-shadow: 0 20px 60px rgba(0,0,0,0.12);
        --card-hover-shadow: 0 30px 80px rgba(0,0,0,0.18);
        --border-radius: 24px;
        --transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Style global avec animation de fond */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Conteneur principal avec effet glassmorphism */
    .main {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        padding: 2rem;
        margin: 1rem;
    }
    
    /* En-tête avec animation */
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        padding: 1.5rem;
        animation: shimmer 3s linear infinite;
        letter-spacing: -1px;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    }
    
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    /* Sous-titre premium */
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Carte glassmorphism premium */
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        padding: 2.5rem;
        box-shadow: var(--card-shadow);
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .result-card:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow: var(--card-hover-shadow);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .result-card:hover::before {
        left: 100%;
    }
    
    /* Badge de statut animé */
    .status-badge {
        display: inline-block;
        padding: 0.8rem 2rem;
        border-radius: 60px;
        font-size: 1.3rem;
        font-weight: 800;
        margin: 1.5rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: 'Poppins', sans-serif;
        animation: pulseGlow 2s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes pulseGlow {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .healthy-badge {
        background: var(--success-gradient);
        color: white;
        box-shadow: 0 8px 25px rgba(56, 239, 125, 0.5);
    }
    
    .unhealthy-badge {
        background: var(--danger-gradient);
        color: white;
        box-shadow: 0 8px 25px rgba(235, 51, 73, 0.5);
    }
    
    /* Barre de confiance améliorée */
    .confidence-bar {
        background: rgba(240, 240, 240, 0.3);
        border-radius: 15px;
        height: 40px;
        margin: 1.5rem 0;
        overflow: hidden;
        position: relative;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 800;
        font-size: 1.1rem;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .confidence-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shine 2s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 200%; }
    }
    
    /* Bouton d'upload premium */
    .upload-section {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border-radius: var(--border-radius);
        padding: 3rem;
        box-shadow: var(--card-shadow);
        border: 3px dashed rgba(102, 126, 234, 0.5);
        text-align: center;
        margin: 2.5rem 0;
        transition: var(--transition);
    }
    
    .upload-section:hover {
        border-color: rgba(102, 126, 234, 0.9);
        background: rgba(255, 255, 255, 0.12);
        transform: scale(1.02);
    }
    
    /* Info box glassmorphism */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        backdrop-filter: blur(15px);
        color: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: var(--transition);
    }
    
    .info-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.6);
    }
    
    /* Animations multiples */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Carte d'instruction premium */
    .instruction-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-left: 6px solid #667eea;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        transition: var(--transition);
    }
    
    .instruction-card:hover {
        transform: translateX(8px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.18);
    }
    
    /* Footer élégant */
    .footer {
        text-align: center;
        padding: 3rem;
        color: rgba(255, 255, 255, 0.7);
        margin-top: 4rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Boutons Streamlit personnalisés */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.8rem 2.5rem;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
        transition: var(--transition);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        font-family: 'Poppins', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
    }
    
    /* File uploader personnalisé */
    .stFileUploader {
        background: transparent;
    }
    
    /* Sidebar moderne */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(20px);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Metrics premium */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Expander premium */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        font-weight: 600;
    }
    
    /* Animation d'entrée pour tous les éléments */
    .element-container {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Amélioration du texte */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.95);
    }
    
    p, li, span {
        font-family: 'Inter', sans-serif;
        line-height: 1.7;
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(56, 239, 125, 0.15);
        border-left: 4px solid #38ef7d;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Error message */
    .stError {
        background: rgba(235, 51, 73, 0.15);
        border-left: 4px solid #eb3349;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Spinner personnalisé */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Scrollbar personnalisée */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* ===== CORRECTION DE LA VISIBILITÉ DES TEXTES ===== */
    
    /* Tous les titres et sous-titres Streamlit */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, 
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: rgba(255, 255, 255, 0.95) !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Tous les paragraphes et textes */
    .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Labels des composants */
    label, .stMarkdown label {
        color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 500 !important;
    }
    
    /* ===== FILE UPLOADER - TEXTE VISIBLE EN NOIR ===== */
    
    /* Zone de drop principale - texte en noir */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px dashed rgba(102, 126, 234, 0.5) !important;
        border-radius: 16px !important;
    }
    
    /* Texte "Drag and drop file here" - NOIR */
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] small,
    [data-testid="stFileUploadDropzone"] p {
        color: #333 !important;
        font-weight: 500 !important;
    }
    
    /* Texte de limite "Limit 200MB per file" - GRIS FONCÉ */
    [data-testid="stFileUploadDropzone"] small {
        color: #666 !important;
    }
    
    /* Label du file uploader */
    [data-testid="stFileUploader"] label {
        color: rgba(255, 255, 255, 0.95) !important;
        font-weight: 500 !important;
    }
    
    /* Bouton "Browse files" */
    [data-testid="stFileUploader"] button {
        color: white !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Nom du fichier uploadé - garder en blanc */
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Container du fichier uploadé */
    [class*="uploadedFile"] {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    [class*="uploadedFile"] * {
        color: rgba(255, 255, 255, 0.95) !important;
    }

    
    /* Texte dans les expanders */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
    }
    
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span,
    [data-testid="stExpander"] div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Headers dans les expanders */
    .streamlit-expanderHeader {
        color: rgba(255, 255, 255, 0.95) !important;
        background: rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Texte dans les colonnes */
    [data-testid="column"] h1,
    [data-testid="column"] h2,
    [data-testid="column"] h3,
    [data-testid="column"] h4,
    [data-testid="column"] p,
    [data-testid="column"] span {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Metric labels et valeurs */
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 1.1rem !important;
    }
    
    [data-testid="stMetricValue"] {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Texte des listes */
    .stMarkdown ul, .stMarkdown ol, .stMarkdown li {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Code et pre */
    code, pre {
        background: rgba(0, 0, 0, 0.3) !important;
        color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 8px;
        padding: 0.2rem 0.4rem;
    }
    
    /* Liens */
    a {
        color: #667eea !important;
        text-decoration: none;
        font-weight: 600;
    }
    
    a:hover {
        color: #764ba2 !important;
        text-decoration: underline;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
        margin: 2rem 0;
    }
    
    /* Texte dans les tabs */
    [data-baseweb="tab"] {
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    [data-baseweb="tab"]:hover {
        color: rgba(255, 255, 255, 1) !important;
    }
    
    /* Captions et small text */
    .caption, small, [data-testid="caption"] {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Chart labels */
    .stPlotlyChart text {
        fill: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Tables */
    table {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    th {
        background: rgba(102, 126, 234, 0.3) !important;
        color: white !important;
        font-weight: 700 !important;
    }
    
    td {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Markdown dans les result-card et instruction-card garde leurs couleurs sombres */
    .result-card h1, .result-card h2, .result-card h3, 
    .result-card h4, .result-card p, .result-card span {
        color: #333 !important;
    }
    
    .instruction-card h1, .instruction-card h2, .instruction-card h3, 
    .instruction-card h4, .instruction-card p, .instruction-card span,
    .instruction-card li {
        color: #333 !important;
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Cache pour charger le modèle une seule fois
@st.cache_resource
def load_ml_model():
    """
    Charge le modèle de Deep Learning hybride.
    Le modèle est mis en cache pour éviter de le recharger à chaque interaction.
    """
    try:
        model_path = "mes models/MODELE_EXPERT_6CLASSES.keras"
        if not os.path.exists(model_path):
            st.error(f"❌ Le fichier du modèle n'a pas été trouvé : {model_path}")
            st.info("Veuillez vérifier que le modèle est présent dans le dossier 'mes models'.")
            return None
        
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """
    Prétraite l'image pour la prédiction.
    
    Args:
        image: Image PIL
        target_size: Tuple (hauteur, largeur) de la taille cible
    
    Returns:
        numpy.ndarray: Image prétraitée prête pour la prédiction
    """
    try:
        # Redimensionner l'image
        img = image.resize(target_size)
        
        # Convertir en array numpy
        img_array = img_to_array(img)
        
        # Normaliser les pixels entre 0 et 1
        img_array = img_array / 255.0
        
        # Ajouter une dimension batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"❌ Erreur lors du prétraitement de l'image : {e}")
        return None

def predict_disease(model, image):
    """
    Effectue une prédiction sur l'image avec classification hiérarchique.
    
    Logique:
    - Classe 0 : Healthy (feuille saine)
    - Classes 1-5 : Unhealthy (pathologies spécifiques)
    
    Args:
        model: Modèle Keras chargé
        image: Image PIL
    
    Returns:
        tuple: (statut_principal, pathologie_specifique, confiance, all_predictions, predicted_class_idx)
    """
    try:
        # Prétraiter l'image
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return None, None, None, None, None
        
        # Faire la prédiction
        predictions = model.predict(processed_image, verbose=0)
        
        # Définition des 6 classes (ordre alphabétique probable)
        class_names = [
            'Healthy',                # 0
            'Red Spider Mite',        # 1
            'Rust Level 1',           # 2
            'Rust Level 2',           # 3
            'Rust Level 3',           # 4
            'Rust Level 4'            # 5
        ]
        
        # Obtenir l'index de la classe avec la plus haute probabilité
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx] * 100
        
        # Déterminer le statut principal (Healthy vs Unhealthy)
        if predicted_class_idx == 0:
            statut_principal = "Healthy"
            pathologie_specifique = None
        else:
            statut_principal = "Unhealthy"
            pathologie_specifique = class_names[predicted_class_idx]
        
        return statut_principal, pathologie_specifique, confidence, predictions[0], predicted_class_idx
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
        return None, None, None, None, None

def display_results(statut_principal, pathologie_specifique, confidence, all_predictions, predicted_class_idx):
    """
    Affiche les résultats de manière hiérarchique avec diagnostic expert.
    
    Args:
        statut_principal: "Healthy" ou "Unhealthy"
        pathologie_specifique: Nom de la pathologie si Unhealthy, None sinon
        confidence: Niveau de confiance (0-100)
        all_predictions: Probabilités pour toutes les 6 classes
        predicted_class_idx: Index de la classe prédite (0-5)
    """
    st.markdown("### 📊 Résultats de l'Analyse Expert")
    
    # Classes pour l'affichage
    class_names = [
        'Healthy (Saine)',
        'Red Spider Mite (Acarien Rouge)',
        'Rust Level 1 (Rouille Niveau 1)',
        'Rust Level 2 (Rouille Niveau 2)',
        'Rust Level 3 (Rouille Niveau 3)',
        'Rust Level 4 (Rouille Niveau 4)'
    ]
    
    # Déterminer le statut et les couleurs
    is_healthy = statut_principal == "Healthy"
    
    # 1. DIAGNOSTIC PRINCIPAL
    if is_healthy:
        badge_class = "healthy-badge"
        icon = "✅"
        status_text = "FEUILLE SAINE"
        message = "La feuille analysée est en bonne santé. Aucune pathologie détectée."
        color = "#38ef7d"
    else:
        badge_class = "unhealthy-badge"
        icon = "⚠️"
        status_text = "FEUILLE MALADE"
        message = f"La feuille présente des signes de maladie."
        color = "#f45c43"
    
    # Carte de diagnostic principal
    st.markdown(f"""
        <div class="result-card">
            <h2 style="text-align: center; margin-bottom: 1rem;">{icon} Diagnostic Principal</h2>
            <div style="text-align: center;">
                <span class="status-badge {badge_class}">{status_text}</span>
            </div>
            <p style="text-align: center; font-size: 1.1rem; color: #666; margin-top: 1rem;">
                {message}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. PATHOLOGIE SPÉCIFIQUE (si malade)
    if not is_healthy and pathologie_specifique:
        # Définir les caractéristiques de chaque pathologie
        pathology_info = {
            'Red Spider Mite': {
                'icon': '🕷️',
                'color': '#f7971e',
                'severity': 'Modérée',
                'description': 'Acarien rouge (Oligonychus coffeae)',
                'symptoms': 'Points jaunes sur les feuilles, toiles fines, dessèchement'
            },
            'Rust Level 1': {
                'icon': '🟡',
                'color': '#ffd93d',
                'severity': 'Légère',
                'description': 'Rouille du caféier - Stade précoce',
                'symptoms': 'Petites taches chlorotiques jaunes sur la face supérieure'
            },
            'Rust Level 2': {
                'icon': '🟠',
                'color': '#ff9800',
                'severity': 'Moyenne',
                'description': 'Rouille du caféier - Stade intermédiaire',                'symptoms': 'Pustules orangées visibles, lésions plus nombreuses'
            },
            'Rust Level 3': {
                'icon': '🔴',
                'color': '#ff5722',
                'severity': 'Sévère',
                'description': 'Rouille du caféier - Stade avancé',
                'symptoms': 'Taches nombreuses et confluentes, défoliation partielle'
            },
            'Rust Level 4': {
                'icon': '🚨',
                'color': '#d32f2f',
                'severity': 'Critique',
                'description': 'Rouille du caféier - Stade critique',
                'symptoms': 'Défoliation sévère, perte massive de feuilles, danger pour la plante'
            }
        }
        
        info = pathology_info.get(pathologie_specifique, {})
        
        st.markdown(f"""
            <div class="result-card" style="border-left: 5px solid {info.get('color', '#f45c43')};">
                <h3 style="color: {info.get('color', '#f45c43')}; margin-top: 0;">
                    {info.get('icon', '🦠')} Pathologie Identifiée
                </h3>
                <div style="background: rgba(255,255,255,0.5); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
                    <h4 style="color: #333; margin-top: 0;">{pathologie_specifique}</h4>
                    <p style="color: #666; margin: 0.5rem 0;">
                        <strong>Description:</strong> {info.get('description', 'Pathologie détectée')}
                    </p>
                    <p style="color: #666; margin: 0.5rem 0;">
                        <strong>Symptômes:</strong> {info.get('symptoms', 'À surveiller')}
                    </p>
                    <p style="color: #666; margin: 0.5rem 0;">
                        <strong>Niveau de sévérité:</strong> 
                        <span style="background: {info.get('color', '#f45c43')}; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-weight: 600;">
                            {info.get('severity', 'Variable')}
                        </span>
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # 3. BARRE DE CONFIANCE
    st.markdown(f"""
        <div class="result-card">
            <h3>🎯 Niveau de Confiance</h3>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence}%; background: linear-gradient(90deg, {color} 0%, {color} 100%);">
                    {confidence:.1f}%
                </div>
            </div>
            <p style="text-align: center; color: #666; margin-top: 0.5rem;">
                Le modèle est confiant à <strong>{confidence:.1f}%</strong> dans ce diagnostic.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 4. DÉTAILS DES PROBABILITÉS
    with st.expander("📈 Voir les détails des probabilités pour toutes les classes"):
        st.markdown("#### Distribution complète des probabilités")
        
        # Afficher sous forme de 2 colonnes (3 classes par colonne)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🌿 Feuille Saine")
            st.metric(
                label=class_names[0],
                value=f"{all_predictions[0] * 100:.2f}%",
                delta=None
            )
            
            st.markdown("##### 🕷️ Acarien")
            st.metric(
                label=class_names[1],
                value=f"{all_predictions[1] * 100:.2f}%",
                delta=None
            )
            
            st.markdown("##### 🟡 Rouille Niveaux 1-2")
            st.metric(
                label=class_names[2],
                value=f"{all_predictions[2] * 100:.2f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                label=class_names[3],
                value=f"{all_predictions[3] * 100:.2f}%",
                delta=None
            )
            
            st.markdown("##### 🔴 Rouille Niveaux 3-4")
            st.metric(
                label=class_names[4],
                value=f"{all_predictions[4] * 100:.2f}%",
                delta=None
            )
            
            st.metric(
                label=class_names[5],
                value=f"{all_predictions[5] * 100:.2f}%",
                delta=None
            )
        
        # Graphique à barres
        st.markdown("##### 📊 Visualisation graphique")
        chart_data = {
            'Classe': class_names,
            'Probabilité (%)': [p * 100 for p in all_predictions]
        }
        st.bar_chart(chart_data, x='Classe', y='Probabilité (%)', color='#667eea')

def main():
    """Fonction principale de l'application"""
    
    # En-tête de l'application
    st.markdown('<h1 class="main-title"> Détection de Maladies des Feuilles de Café</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Système Expert d\'Analyse par Intelligence Artificielle - Classification Hiérarchique</p>', unsafe_allow_html=True)
    
    # Barre latérale avec informations
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/coffee-beans-.png", width=100)
        st.markdown("## 📚 À propos")
        st.markdown("""
        Cette application utilise un **modèle expert de Deep Learning** 
        pour identifier les pathologies des feuilles de café Robusta.
        
        ### 🎯 Système de Classification
        
        **Niveau 1 - Diagnostic Principal:**
        - ✅ Healthy (Saine)
        - ⚠️ Unhealthy (Malade)
        
        **Niveau 2 - Identification de la Pathologie:**
        - 🕷️ Red Spider Mite (Acarien Rouge)
        - 🟡 Rust Level 1 (Rouille Légère)
        - 🟠 Rust Level 2 (Rouille Moyenne)
        - 🔴 Rust Level 3 (Rouille Sévère)
        - 🚨 Rust Level 4 (Rouille Critique)
        
        ### 🔬 Modèle
        - **Architecture:** CNN avancé multi-classe
        - **Classes:** 6 catégories (1 saine + 5 pathologies)
        - **Dataset:** RoCoLE (Robusta Coffee Leaf images)
        - **Input:** Images 224x224 pixels
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Instructions")
        st.markdown("""
        1. **Téléchargez** une image de feuille de café
        2. **Attendez** l'analyse automatique
        3. **Consultez** les résultats du diagnostic
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Technologies")
        st.markdown("""
        - **TensorFlow/Keras** - Deep Learning Framework
        - **Streamlit** - Interface Web Interactive
        - **CNN Multi-Classe** - Architecture Deep Learning
        - **NumPy/PIL** - Traitement d'Images
        """)
        
        st.markdown("---")
        st.markdown("### 👥 Équipe")
        st.markdown("**Groupe 8** - Master 2 Data Science UFHB")
        st.markdown("Projet de Deep Learning")
    
    # Zone principale
    st.markdown("""
        <div class="info-box">
            <h3 style="margin-top: 0;">ℹ️ Comment utiliser ce système expert ?</h3>
            <p style="margin-bottom: 0;">
                Téléchargez une photo claire d'une feuille de café Robusta. 
                Notre système expert analysera l'image et établira un diagnostic hiérarchique :
                d'abord le statut général (saine/malade), puis si malade, identifiera la pathologie spécifique 
                parmi les 5 maladies connues. Assurez-vous que la feuille est bien visible et l'image nette.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Charger le modèle
    with st.spinner("🔄 Chargement du modèle..."):
        model = load_ml_model()
    
    if model is None:
        st.error("❌ Impossible de charger le modèle. Veuillez vérifier l'installation.")
        st.stop()
    
    st.success("✅ Modèle chargé avec succès !")
    
    # Section d'upload
    st.markdown("---")
    st.markdown("## 📤 Télécharger une Image")
    
    uploaded_file = st.file_uploader(
        "Choisissez une image de feuille de café (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Formats supportés: JPG, JPEG, PNG. Taille maximale: 200MB"
    )
    
    if uploaded_file is not None:
        # Lire et afficher l'image
        try:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Afficher l'image téléchargée
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### 📸 Image téléchargée")
                st.image(image, caption="Image de la feuille à analyser", use_container_width=True)
            
            # Bouton d'analyse
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                analyze_button = st.button(
                    "🔬 Analyser l'Image",
                    type="primary",
                    use_container_width=True
                )
            
            if analyze_button:
                # Analyser l'image
                with st.spinner("🔍 Analyse en cours..."):
                    statut_principal, pathologie_specifique, confidence, all_predictions, predicted_class_idx = predict_disease(model, image)
                
                if statut_principal is not None:
                    st.markdown("---")
                    # Afficher les résultats
                    display_results(statut_principal, pathologie_specifique, confidence, all_predictions, predicted_class_idx)
                    
                    # Recommandations spécifiques
                    st.markdown("---")
                    st.markdown("### 💡 Recommandations d'Actions")
                    
                    if statut_principal == "Healthy":
                        st.markdown("""
                            <div class="instruction-card">
                                <h4 style="color: #38ef7d; margin-top: 0;">✅ Feuille Saine Détectée</h4>
                                <ul>
                                    <li><strong>Surveillance préventive:</strong> Continuez les pratiques agricoles actuelles</li>
                                    <li><strong>Contrôle régulier:</strong> Inspectez les plants chaque semaine</li>
                                    <li><strong>Nutrition:</strong> Maintenez un programme de fertilisation équilibré</li>
                                    <li><strong>Prophylaxie:</strong> Appliquez des traitements préventifs si nécessaire</li>
                                    <li><strong>Documentation:</strong> Notez l'état actuel pour référence future</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Recommandations spécifiques selon la pathologie
                        recommendations = {
                            'Red Spider Mite': {
                                'color': '#f7971e',
                                'icon': '🕷️',
                                'title': 'Acarien Rouge Détecté',
                                'actions': [
                                    '<strong>Action immédiate:</strong> Isoler les plants infectés',
                                    '<strong>Traitement acaricide:</strong> Appliquer un acaricide spécifique (ex: abamectine, spiromesifen)',
                                    '<strong>Contrôle biologique:</strong> Introduire des prédateurs naturels (acariens prédateurs)',
                                    '<strong>Gestion environnementale:</strong> Augmenter l\'humidité relative (> 70%)',
                                    '<strong>Éviter:</strong> La sur-fertilisation azotée qui favorise les acariens',
                                    '<strong>Monitoring:</strong> Surveiller hebdomadairement avec des loupes',
                                    '<strong>Prévention:</strong> Éliminer les mauvaises herbes environnantes'
                                ]
                            },
                            'Rust Level 1': {
                                'color': '#ffd93d',
                                'icon': '🟡',
                                'title': 'Rouille Niveau 1 - Intervention Précoce',
                                'actions': [
                                    '<strong>Chance de contrôle:</strong> Excellent! Intervention au stade précoce',
                                    '<strong>Fongicide systémique:</strong> Appliquer triazole ou strobilurine',
                                    '<strong>Action rapide:</strong> Traiter sous 48h pour éviter la progression',
                                    '<strong>Élimination:</strong> Retirer et brûler les feuilles légèrement affectées',
                                    '<strong>Espacement:</strong> Améliorer la circulation d\'air entre les plants',
                                    '<strong>Nutrition:</strong> Renforcer avec potassium et micronutriments',
                                    '<strong>Surveillance:</strong> Inspections quotidiennes pendant 2 semaines'
                                ]
                            },
                            'Rust Level 2': {
                                'color': '#ff9800',
                                'icon': '🟠',
                                'title': 'Rouille Niveau 2 - Action Urgente Requise',
                                'actions': [
                                    '<strong>Urgence:</strong> Traitement fongicide dans les 24h',
                                    '<strong>Protocole intensif:</strong> Fongicide à base de cuivre + triazole',
                                    '<strong>Double application:</strong> Répéter le traitement après 10-14 jours',
                                    '<strong>Défoliation ciblée:</strong> Enlever les feuilles moyennement à fortement infectées',
                                    '<strong>Quarantaine:</strong> Isoler immédiatement la zone affectée',
                                    '<strong>Réduire humidité:</strong> Éviter l\'irrigation par aspersion',
                                    '<strong>Consultation:</strong> Faire appel à un phytopathologiste',
                                    '<strong>Traçabilité:</strong> Cartographier les zones infectées'
                                ]
                            },
                            'Rust Level 3': {
                                'color': '#ff5722',
                                'icon': '🔴',
                                'title': 'Rouille Niveau 3 - Situation Critique',
                                'actions': [
                                    '<strong>⚠️ ALERTE CRITIQUE:</strong> Intervention d\'urgence requise',
                                    '<strong>Traitement agressif:</strong> Fongicide systémique à dose maximale',
                                    '<strong>Applications fréquentes:</strong> Traiter tous les 7 jours pendant 1 mois',
                                    '<strong>Défoliation majeure:</strong> Retirer jusqu\'à 60% des feuilles infectées',
                                    '<strong>Tailler:</strong> Élaguer les branches fortement atteintes',
                                    '<strong>Zone tampon:</strong> Traiter aussi les plants dans un rayon de 10m',
                                    '<strong>Mesures drastiques:</strong> Envisager l\'arrachage des plants les plus atteints',
                                    '<strong>Expert obligatoire:</strong> Consultation immédiate d\'un agronome',
                                    '<strong>Perte de rendement:</strong> Prévoir 30-50% de baisse de production'
                                ]
                            },
                            'Rust Level 4': {
                                'color': '#d32f2f',
                                'icon': '🚨',
                                'title': 'Rouille Niveau 4 - URGENCE MAXIMALE',
                                'actions': [
                                    '<strong>🚨 DANGER IMMINENT:</strong> Risque de perte totale du plant',
                                    '<strong>Décision urgente:</strong> Évaluer viabilité du plant (< 30% feuilles saines = arracher)',
                                    '<strong>Si maintien:</strong> Traitement fongicide + nutritionnel intensif',
                                    '<strong>Défoliation complète:</strong> Retirer TOUTES les feuilles infectées',
                                    '<strong>Taille sévère:</strong> Rabattre au niveau du tronc si nécessaire',
                                    '<strong>Quarantaine stricte:</strong> Isoler avec barrière physique',
                                    '<strong>Protection zone saine:</strong> Traiter préventivement tous les plants dans 20m',
                                    '<strong>Désinfection:</strong> Désinfecter tous les outils après usage',
                                    '<strong>Arrachage possible:</strong> Détruire le plant si l\'infection progresse',
                                    '<strong>Réglementation:</strong> Déclarer aux autorités phytosanitaires si requis',
                                    '<strong>Perte économique:</strong> Anticiper perte de 70-100% du rendement'
                                ]
                            }
                        }
                        
                        reco = recommendations.get(pathologie_specifique, {
                            'color': '#f45c43',
                            'icon': '⚠️',
                            'title': 'Feuille Malade Détectée',
                            'actions': [
                                '<strong>Action immédiate:</strong> Isoler les plants affectés',
                                'Consulter un agronome spécialisé',
                                'Analyser les conditions environnementales',
                                'Surveiller la propagation'
                            ]
                        })
                        
                        actions_html = ''.join([f'<li>{action}</li>' for action in reco['actions']])
                        
                        st.markdown(f"""
                            <div class="instruction-card" style="border-left: 5px solid {reco['color']};">
                                <h4 style="color: {reco['color']}; margin-top: 0;">{reco['icon']} {reco['title']}</h4>
                                <ul>
                                    {actions_html}
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("❌ Erreur lors de l'analyse de l'image.")
        
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement de l'image : {e}")
    
    else:
        # Instructions quand aucune image n'est téléchargée
        st.markdown("""
            <div class="upload-section">
                <h2 style="color: rgba(255, 255, 255, 0.9); margin-top: 0; font-size: 2.5rem;">📁 Glissez votre image ici</h2>
                <p style="color: rgba(255, 255, 255, 0.8); margin: 1.5rem 0; font-size: 1.2rem;">
                    Sélectionnez une image de feuille de café pour commencer l'analyse intelligente
                </p>
                <div style="margin: 2rem 0;">
                    <span style="display: inline-block; background: rgba(102, 126, 234, 0.2); color: rgba(255, 255, 255, 0.9); padding: 0.5rem 1.5rem; border-radius: 50px; margin: 0.5rem; font-weight: 600;">JPG</span>
                    <span style="display: inline-block; background: rgba(102, 126, 234, 0.2); color: rgba(255, 255, 255, 0.9); padding: 0.5rem 1.5rem; border-radius: 50px; margin: 0.5rem; font-weight: 600;">JPEG</span>
                    <span style="display: inline-block; background: rgba(102, 126, 234, 0.2); color: rgba(255, 255, 255, 0.9); padding: 0.5rem 1.5rem; border-radius: 50px; margin: 0.5rem; font-weight: 600;">PNG</span>
                </div>
                <p style="color: rgba(255, 255, 255, 0.6); font-size: 0.95rem; margin-top: 1.5rem;">
                    ✨ Taille maximale : 200MB | 🔒 Traitement sécurisé local
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <p style="margin: 0.5rem 0;">
                <strong>Projet de Deep Learning</strong> - Master 2 Data Science UFHB
            </p>
            <p style="margin: 0.5rem 0; color: #999;">
                Groupe 8 - Système Expert de Classification des Maladies du Café Robusta
            </p>
            <p style="margin: 0.5rem 0; color: #999; font-size: 0.9rem;">
                Modèle CNN Expert 6 Classes | Diagnostic Hiérarchique Healthy vs Pathologies
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
