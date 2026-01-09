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
    
    /* Texte du file uploader */
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Section file uploader */
    [data-testid="stFileUploader"] {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Nom du fichier uploadé - toutes les variantes possibles */
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploadDropzone"] {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Spécifiquement le nom du fichier */
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"],
    [data-testid="stFileUploader"] > div > div {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Bouton browse files */
    [data-testid="stFileUploader"] button {
        color: #333 !important;
        background: rgba(255, 255, 255, 0.9) !important;
        border: none !important;
        font-weight: 600 !important;
    }
    
    /* Info text dans file uploader */
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] small {
        color: rgba(255, 255, 255, 0.85) !important;
    }
    
    /* RÈGLE UNIVERSELLE - Force TOUS les textes du file uploader en blanc */
    [data-testid="stFileUploader"] *:not(button) {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* Container principal du fichier uploadé */
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
        model_path = "mes models/modele_hybride_final.keras"
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
    Effectue une prédiction sur l'image.
    
    Args:
        model: Modèle Keras chargé
        image: Image PIL
    
    Returns:
        tuple: (classe_prédite, confiance)
    """
    try:
        # Prétraiter l'image
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return None, None
        
        # Faire la prédiction
        predictions = model.predict(processed_image, verbose=0)
        
        # Classes : 0 = Healthy (Saine), 1 = Unhealthy (Malade)
        class_names = ['Healthy (Saine)', 'Unhealthy (Malade)']
        
        # Obtenir l'index de la classe avec la plus haute probabilité
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx] * 100
        
        predicted_class = class_names[predicted_class_idx]
        
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
        return None, None, None

def display_results(predicted_class, confidence, all_predictions):
    """
    Affiche les résultats de la prédiction de manière élégante.
    
    Args:
        predicted_class: Nom de la classe prédite
        confidence: Niveau de confiance (0-100)
        all_predictions: Probabilités pour toutes les classes
    """
    st.markdown("### 📊 Résultats de l'Analyse")
    
    # Déterminer si la feuille est saine ou malade
    is_healthy = 'Healthy' in predicted_class
    
    # Afficher le statut avec un badge coloré
    if is_healthy:
        badge_class = "healthy-badge"
        icon = "✅"
        status_text = "FEUILLE SAINE"
        message = "La feuille analysée est en bonne santé."
        color = "#38ef7d"
    else:
        badge_class = "unhealthy-badge"
        icon = "⚠️"
        status_text = "FEUILLE MALADE"
        message = "La feuille analysée présente des signes de maladie."
        color = "#f45c43"
    
    # Carte de résultat
    st.markdown(f"""
        <div class="result-card">
            <h2 style="text-align: center; margin-bottom: 1rem;">{icon} Diagnostic</h2>
            <div style="text-align: center;">
                <span class="status-badge {badge_class}">{status_text}</span>
            </div>
            <p style="text-align: center; font-size: 1.1rem; color: #666; margin-top: 1rem;">
                {message}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Barre de confiance
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
    
    # Détails des probabilités
    with st.expander("📈 Voir les détails des probabilités"):
        class_names = ['Healthy (Saine)', 'Unhealthy (Malade)']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌿 Feuille Saine")
            st.metric(
                label="Probabilité",
                value=f"{all_predictions[0] * 100:.2f}%",
                delta=None
            )
        
        with col2:
            st.markdown("#### 🦠 Feuille Malade")
            st.metric(
                label="Probabilité",
                value=f"{all_predictions[1] * 100:.2f}%",
                delta=None
            )
        
        # Graphique à barres
        st.markdown("##### Distribution des probabilités")
        chart_data = {
            'Classe': class_names,
            'Probabilité (%)': [all_predictions[0] * 100, all_predictions[1] * 100]
        }
        st.bar_chart(chart_data, x='Classe', y='Probabilité (%)', color='#667eea')

def main():
    """Fonction principale de l'application"""
    
    # En-tête de l'application
    st.markdown('<h1 class="main-title"> Détection de Maladies des Feuilles de Café</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Analyse automatique par Intelligence Artificielle - Modèle Hybride CNN</p>', unsafe_allow_html=True)
    
    # Barre latérale avec informations
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/coffee-beans-.png", width=100)
        st.markdown("## 📚 À propos")
        st.markdown("""
        Cette application utilise un **modèle de Deep Learning hybride** 
        combinant :
        - **CNN avancé** (MobileNetV2)
        - **Autoencodeur** pour la détection d'anomalies
        
        Le modèle a été entraîné pour distinguer les feuilles saines 
        des feuilles malades.
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
        - **TensorFlow/Keras** - Deep Learning
        - **Streamlit** - Interface Web
        - **MobileNetV2** - Architecture CNN
        - **Autoencodeur** - Détection d'anomalies
        """)
        
        st.markdown("---")
        st.markdown("### 👥 Équipe")
        st.markdown("**Groupe 8** - Master 2 Data Science UFHB")
        st.markdown("Projet de Deep Learning")
    
    # Zone principale
    st.markdown("""
        <div class="info-box">
            <h3 style="margin-top: 0;">ℹ️ Comment utiliser cette application ?</h3>
            <p style="margin-bottom: 0;">
                Téléchargez une photo claire d'une feuille de café. 
                Notre modèle d'IA analysera l'image et vous indiquera si la feuille est saine ou malade.
                Pour de meilleurs résultats, assurez-vous que la feuille est bien visible et que l'image est nette.
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
                    predicted_class, confidence, all_predictions = predict_disease(model, image)
                
                if predicted_class is not None:
                    st.markdown("---")
                    # Afficher les résultats
                    display_results(predicted_class, confidence, all_predictions)
                    
                    # Recommandations
                    st.markdown("---")
                    st.markdown("### 💡 Recommandations")
                    
                    if 'Healthy' in predicted_class:
                        st.markdown("""
                            <div class="instruction-card">
                                <h4 style="color: #38ef7d; margin-top: 0;">✅ Feuille Saine Détectée</h4>
                                <ul>
                                    <li>Continuez les pratiques agricoles actuelles</li>
                                    <li>Maintenez une surveillance régulière</li>
                                    <li>Assurez une nutrition adéquate des plants</li>
                                    <li>Vérifiez régulièrement l'état des autres feuilles</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class="instruction-card">
                                <h4 style="color: #f45c43; margin-top: 0;">⚠️ Feuille Malade Détectée</h4>
                                <ul>
                                    <li><strong>Action immédiate requise :</strong> Isolez les plants affectés</li>
                                    <li>Consultez un agronome spécialisé</li>
                                    <li>Analysez les conditions environnementales (humidité, température)</li>
                                    <li>Envisagez un traitement approprié</li>
                                    <li>Surveillez la propagation aux plants voisins</li>
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
                Groupe 8 - Classification des Maladies des Feuilles de Café
            </p>
            <p style="margin: 0.5rem 0; color: #999; font-size: 0.9rem;">
                Modèle Hybride CNN + Autoencodeur | Précision optimisée pour la détection
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
