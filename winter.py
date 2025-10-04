import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
from scipy.stats import norm, shapiro, kurtosis, skew
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.ticker import FuncFormatter
import io

warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(
    page_title="Analyse Série Temporelle - Holt-Winters",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour formater les nombres
def format_number(value):
    """Formate les nombres avec séparateurs de milliers"""
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float)):
        if abs(value) >= 1e12:
            return f"{value:,.0f}"
        elif abs(value) >= 1e9:
            return f"{value:,.2f}"
        elif abs(value) >= 1:
            return f"{value:,.2f}"
        else:
            return f"{value:.6f}"
    return str(value)

# CSS personnalisé pour le design sombre et moderne
st.markdown("""
<style>
    /* Background avec effet technologique */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        background-attachment: fixed;
    }
    
    /* Effet de grille en arrière-plan */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Animation de pulsation pour les titres */
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 20px rgba(0, 217, 255, 0.5), 0 0 30px rgba(0, 217, 255, 0.3); }
        50% { text-shadow: 0 0 30px rgba(0, 217, 255, 0.8), 0 0 40px rgba(0, 217, 255, 0.5); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Titres personnalisés */
    h1 {
        color: #00d9ff !important;
        animation: glow 2s ease-in-out infinite;
        font-weight: 800 !important;
        letter-spacing: 2px;
    }
    
    h2 {
        color: #00ffaa !important;
        text-shadow: 0 0 15px rgba(0, 255, 170, 0.4);
        font-weight: 700 !important;
        animation: slideIn 0.5s ease-out;
    }
    
    h3 {
        color: #4dd4ff !important;
        font-weight: 600 !important;
        text-shadow: 0 0 10px rgba(77, 212, 255, 0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f3a 100%);
        border-right: 2px solid rgba(0, 217, 255, 0.3);
    }
    
    /* Metrics cards avec animation */
    [data-testid="stMetricValue"] {
        color: #00d9ff !important;
        font-size: 32px !important;
        font-weight: 700 !important;
        text-shadow: 0 0 15px rgba(0, 217, 255, 0.6);
        font-family: 'Courier New', monospace !important;
        animation: slideIn 0.8s ease-out;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetricDelta"] {
        color: #00ffaa !important;
    }
    
    /* Container des metrics avec effet */
    [data-testid="metric-container"] {
        background: rgba(26, 31, 58, 0.6);
        border: 1px solid rgba(0, 217, 255, 0.3);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 5px 15px rgba(0, 217, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        border-color: rgba(0, 217, 255, 0.6);
        box-shadow: 0 8px 25px rgba(0, 217, 255, 0.3);
        transform: translateY(-2px);
    }
    
    /* Boutons */
    .stButton button {
        background: linear-gradient(135deg, #00d9ff 0%, #00ffaa 100%);
        color: #0a0e27;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        box-shadow: 0 5px 15px rgba(0, 217, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 217, 255, 0.6);
    }
    
    /* Tables avec effet néon */
    [data-testid="stDataFrame"] {
        background: rgba(15, 20, 25, 0.9) !important;
        border: 2px solid rgba(0, 217, 255, 0.5);
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.2);
        animation: slideIn 0.6s ease-out;
    }
    
    /* Style des cellules de tableau */
    .dataframe {
        color: #ffffff !important;
        font-family: 'Courier New', monospace;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #00d9ff 0%, #00ffaa 100%) !important;
        color: #0a0e27 !important;
        font-weight: 700 !important;
        padding: 12px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: none !important;
    }
    
    .dataframe td {
        background: rgba(26, 31, 58, 0.6) !important;
        color: #00d9ff !important;
        padding: 10px !important;
        border-bottom: 1px solid rgba(0, 217, 255, 0.2) !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    .dataframe tr:hover td {
        background: rgba(0, 217, 255, 0.1) !important;
        transition: all 0.3s ease;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(26, 31, 58, 0.5);
        border: 2px dashed rgba(0, 217, 255, 0.4);
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Divider */
    hr {
        border: 1px solid rgba(0, 217, 255, 0.2);
    }
    
    /* Info boxes avec effet néon */
    .stAlert {
        background: rgba(0, 217, 255, 0.15) !important;
        border-left: 4px solid #00d9ff;
        border-radius: 10px;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.2);
        animation: slideIn 0.6s ease-out;
    }
    
    /* Success box */
    .stSuccess {
        background: rgba(0, 255, 170, 0.15) !important;
        border-left: 4px solid #00ffaa !important;
        box-shadow: 0 0 20px rgba(0, 255, 170, 0.2);
    }
    
    /* Warning box */
    .stWarning {
        background: rgba(255, 165, 0, 0.15) !important;
        border-left: 4px solid #ffa500 !important;
        box-shadow: 0 0 20px rgba(255, 165, 0, 0.2);
    }
    
    /* Error box */
    .stError {
        background: rgba(255, 0, 102, 0.15) !important;
        border-left: 4px solid #ff0066 !important;
        box-shadow: 0 0 20px rgba(255, 0, 102, 0.2);
    }
    
    /* Texte dans les boxes */
    .stAlert p, .stSuccess p, .stWarning p, .stError p {
        color: #ffffff !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background: rgba(26, 31, 58, 0.6);
        border: 1px solid rgba(0, 217, 255, 0.3);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger les données
@st.cache_data
def load_data(file, file_type):
    try:
        if file_type == "csv":
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return None

# Fonction pour nettoyer et préparer les données
def prepare_timeseries(df, date_col, value_col, freq='Y'):
    df_clean = df[[date_col, value_col]].copy()
    
    # Nettoyer les valeurs
    if df_clean[value_col].dtype == 'object':
        df_clean[value_col] = df_clean[value_col].str.replace('\xa0', '', regex=True)
        df_clean[value_col] = df_clean[value_col].str.replace(' ', '', regex=True)
        df_clean[value_col] = pd.to_numeric(df_clean[value_col], errors='coerce')
    
    # Créer l'index temporel
    df_clean['Date_Index'] = pd.date_range(start=str(df_clean[date_col].iloc[0]), 
                                            periods=len(df_clean), freq=freq)
    df_clean.set_index('Date_Index', inplace=True)
    
    return df_clean

# Titre principal
st.markdown("<h1 style='text-align: center;'>📊 ANALYSE DE SÉRIE TEMPORELLE</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #a0aec0;'>Méthode de Lissage Exponentiel Holt-Winters</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ CONFIGURATION")
    st.markdown("---")
    
    # Upload fichier
    uploaded_file = st.file_uploader(
        "📁 Charger vos données",
        type=['csv', 'xlsx', 'xls'],
        help="Format supporté: CSV, Excel (XLSX, XLS)"
    )
    
    if uploaded_file:
        file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'excel'
        df = load_data(uploaded_file, file_type)
        
        if df is not None:
            st.success("✅ Fichier chargé avec succès!")
            
            st.markdown("### 🎯 Paramètres")
            
            # Sélection des colonnes
            date_col = st.selectbox("📅 Colonne Date/Année", df.columns)
            value_col = st.selectbox("💰 Colonne Valeur", df.columns)
            
            # Fréquence
            freq_options = {
                'Annuelle': 'Y',
                'Trimestrielle': 'Q',
                'Mensuelle': 'M',
                'Hebdomadaire': 'W',
                'Quotidienne': 'D'
            }
            freq_label = st.selectbox("📆 Fréquence", list(freq_options.keys()))
            freq = freq_options[freq_label]
            
            # Nombre de prévisions
            n_forecast = st.slider("🔮 Périodes à prévoir", 1, 20, 4)
            
            st.markdown("---")
            analyze_btn = st.button("🚀 LANCER L'ANALYSE", use_container_width=True)

# Corps principal
if uploaded_file and df is not None:
    
    # Préparer les données
    try:
        ts_data = prepare_timeseries(df, date_col, value_col, freq)
        series = ts_data[value_col].dropna()
    except Exception as e:
        st.error(f"Erreur lors de la préparation des données : {e}")
        st.stop()
    
    if 'analyze_btn' in locals() and analyze_btn:
        
        # TAB NAVIGATION
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Exploration", 
            "📈 Statistiques", 
            "🔄 Lissage", 
            "🔍 Validation", 
            "🔮 Prévision"
        ])
        
        # TAB 1: EXPLORATION
        with tab1:
            st.markdown("## 📊 EXPLORATION DES DONNÉES")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📏 Observations", f"{len(series):,}")
            with col2:
                st.metric("📅 Début", series.index[0].strftime('%Y'))
            with col3:
                st.metric("📅 Fin", series.index[-1].strftime('%Y'))
            with col4:
                missing = series.isna().sum()
                st.metric("❌ Valeurs manquantes", f"{missing:,}")
            
            st.markdown("---")
            
            # Aperçu des données
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📋 Aperçu des données")
                # Formater le dataframe pour l'affichage
                display_df = ts_data.head(10).copy()
                display_df[value_col] = display_df[value_col].apply(lambda x: format_number(x))
                st.dataframe(display_df, use_container_width=True, height=400)
            
            with col2:
                st.markdown("### 📋 Statistiques rapides")
                stats_df = series.describe().to_frame()
                stats_df.columns = ['Valeur']
                stats_df['Valeur'] = stats_df['Valeur'].apply(lambda x: format_number(x))
                st.dataframe(stats_df, use_container_width=True, height=400)
            
            st.markdown("---")
            
            # Graphique série temporelle
            st.markdown("### 📈 Évolution de la série temporelle")
            fig, ax = plt.subplots(figsize=(14, 6), facecolor='#0a0e27')
            ax.set_facecolor('#1a1f3a')
            ax.plot(series.index, series, color='#00d9ff', linewidth=2.5, marker='o', markersize=5)
            
            # Ajouter des annotations pour chaque décennie
            start_year = series.index.min().year
            end_year = series.index.max().year
            
            for year in range(start_year, end_year + 1):
                if year % 10 == 0:  # Tous les 10 ans
                    year_data = series[series.index.year == year]
                    if not year_data.empty:
                        last_value = year_data.iloc[-1]
                        last_date = year_data.index[-1]
                        ax.annotate(f'{year}\n{last_value:,.0f}',
                                   xy=(last_date, last_value),
                                   xytext=(0, 15), 
                                   textcoords='offset points',
                                   ha='center', 
                                   fontsize=9, 
                                   color='#00ffaa',
                                   bbox=dict(boxstyle='round,pad=0.4', 
                                           facecolor='#1a1f3a', 
                                           edgecolor='#00ffaa',
                                           alpha=0.9),
                                   fontweight='bold')
            
            # Formatter l'axe Y pour afficher les nombres complets
            def format_y_axis(value, pos):
                if abs(value) >= 1e12:
                    return f'{value/1e12:.1f}T'
                elif abs(value) >= 1e9:
                    return f'{value/1e9:.1f}B'
                elif abs(value) >= 1e6:
                    return f'{value/1e6:.1f}M'
                else:
                    return f'{value:,.0f}'
            
            ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
            
            ax.set_title("Évolution de la série temporelle", color='white', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Date", color='#a0aec0', fontsize=12)
            ax.set_ylabel("Valeur", color='#a0aec0', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3, color='#4dd4ff')
            ax.tick_params(colors='#a0aec0')
            for spine in ax.spines.values():
                spine.set_color('#4dd4ff')
                spine.set_linewidth(2)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Histogramme et Boxplot
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Distribution")
                fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0a0e27')
                ax.set_facecolor('#1a1f3a')
                ax.hist(series, bins=20, color='#00d9ff', edgecolor='#00ffaa', alpha=0.7)
                ax.set_title("Distribution des valeurs", color='white', fontsize=14, fontweight='bold')
                ax.set_xlabel("Valeur", color='#a0aec0')
                ax.set_ylabel("Fréquence", color='#a0aec0')
                ax.grid(True, alpha=0.3, color='#4dd4ff')
                ax.tick_params(colors='#a0aec0')
                for spine in ax.spines.values():
                    spine.set_color('#4dd4ff')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown("### 📦 Boxplot")
                fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0a0e27')
                ax.set_facecolor('#1a1f3a')
                bp = ax.boxplot(series, vert=True, patch_artist=True,
                                boxprops=dict(facecolor='#00d9ff', color='#00ffaa'),
                                whiskerprops=dict(color='#00ffaa'),
                                capprops=dict(color='#00ffaa'),
                                medianprops=dict(color='#ff0066', linewidth=2),
                                flierprops=dict(marker='o', markerfacecolor='#ff0066', markersize=6, alpha=0.5))
                ax.set_title("Distribution (Boxplot)", color='white', fontsize=14, fontweight='bold')
                ax.set_ylabel("Valeur", color='#a0aec0')
                ax.grid(axis='y', linestyle='--', alpha=0.3, color='#4dd4ff')
                ax.tick_params(colors='#a0aec0')
                for spine in ax.spines.values():
                    spine.set_color('#4dd4ff')
                plt.tight_layout()
                st.pyplot(fig)
        
        # TAB 2: STATISTIQUES
        with tab2:
            st.markdown("## 📈 INDICES STATISTIQUES")
            
            # Indices de tendance centrale
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Moyenne", format_number(series.mean()))
            with col2:
                st.metric("📊 Médiane", format_number(series.median()))
            with col3:
                st.metric("📊 Écart-type", format_number(series.std()))
            
            st.markdown("---")
            
            # Indices de forme
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Variance", format_number(series.var()))
            with col2:
                kurtosis_val = kurtosis(series, bias=False)
                st.metric("📊 Kurtosis", f"{kurtosis_val:.4f}")
            with col3:
                skewness_val = skew(series, bias=False)
                st.metric("📊 Skewness", f"{skewness_val:.4f}")
            
            st.markdown("---")
            
            # ACF et PACF
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔗 Autocorrélation (ACF)")
                fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0a0e27')
                ax.set_facecolor('#1a1f3a')
                plot_acf(series.dropna(), lags=10, alpha=0.05, ax=ax, color='#00d9ff')
                
                # Ajouter les valeurs sur le graphique ACF
                acf_values = acf(series.dropna(), nlags=10, fft=False)
                for i, v in enumerate(acf_values):
                    ax.text(i, v + 0.03, f"{v:.3f}", 
                           ha='center', va='bottom', 
                           fontsize=8, color='#00ffaa', 
                           fontweight='bold')
                
                ax.set_title("Fonction d'autocorrélation (ACF)", color='white', fontsize=14, fontweight='bold')
                ax.set_xlabel("Lag", color='#a0aec0')
                ax.set_ylabel("Autocorrélation", color='#a0aec0')
                ax.tick_params(colors='#a0aec0')
                for spine in ax.spines.values():
                    spine.set_color('#4dd4ff')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown("### 🔗 Autocorrélation Partielle (PACF)")
                fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0a0e27')
                ax.set_facecolor('#1a1f3a')
                plot_pacf(series.dropna(), lags=10, alpha=0.05, ax=ax, method='ywm', color='#ff0066')
                
                # Ajouter les valeurs sur le graphique PACF
                pacf_values = pacf(series.dropna(), nlags=10, method='ywm')
                for i, v in enumerate(pacf_values):
                    ax.text(i, v + 0.03, f"{v:.3f}",
                           ha='center', va='bottom',
                           fontsize=8, color='#00ffaa',
                           fontweight='bold')
                
                ax.set_title("Fonction d'autocorrélation partielle (PACF)", color='white', fontsize=14, fontweight='bold')
                ax.set_xlabel("Lag", color='#a0aec0')
                ax.set_ylabel("PACF", color='#a0aec0')
                ax.tick_params(colors='#a0aec0')
                for spine in ax.spines.values():
                    spine.set_color('#4dd4ff')
                plt.tight_layout()
                st.pyplot(fig)
        
        # TAB 3: LISSAGE
        with tab3:
            st.markdown("## 🔄 LISSAGE EXPONENTIEL HOLT-WINTERS")
            
            with st.spinner("🔄 Calcul du modèle en cours..."):
                # Ajuster le modèle
                model = ExponentialSmoothing(
                    series,
                    trend='add',
                    seasonal=None,
                    initialization_method='estimated'
                )
                fitted_model = model.fit()
                
                # Paramètres du modèle
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Alpha (niveau)", f"{fitted_model.params['smoothing_level']:.5f}")
                with col2:
                    st.metric("📈 Beta (tendance)", f"{fitted_model.params['smoothing_trend']:.5f}")
                with col3:
                    st.metric("❌ SSE", format_number(fitted_model.sse))
                
                st.markdown("---")
                
                # Graphique de comparaison
                st.markdown("### 📊 Série Originale vs Série Lissée")
                fig, ax = plt.subplots(figsize=(14, 6), facecolor='#0a0e27')
                ax.set_facecolor('#1a1f3a')
                ax.plot(series.index, series, color='#00d9ff', linewidth=2.5, marker='o', 
                        markersize=5, label='Série originale', alpha=0.8)
                ax.plot(series.index, fitted_model.fittedvalues, color='#ff0066', linewidth=2.5, 
                        marker='s', markersize=4, label='Série lissée (Holt)', alpha=0.9)
                
                # Ajouter des annotations pour chaque décennie
                start_year = series.index.min().year
                end_year = series.index.max().year
                
                for year in range(start_year, end_year + 1):
                    if year % 10 == 0:  # Tous les 10 ans
                        year_mask = series.index.year == year
                        if year_mask.sum() > 0:
                            try:
                                # Annotation pour la série originale
                                last_orig_value = float(series[year_mask].iloc[-1])
                                last_orig_date = series[year_mask].index[-1]
                                ax.annotate(f'{year}\n{last_orig_value:,.0f}',
                                           xy=(last_orig_date, last_orig_value),
                                           xytext=(0, 15), 
                                           textcoords='offset points',
                                           ha='center', 
                                           fontsize=9, 
                                           color='#00d9ff',
                                           bbox=dict(boxstyle='round,pad=0.3', 
                                                   facecolor='#1a1f3a', 
                                                   edgecolor='#00d9ff',
                                                   alpha=0.9),
                                           fontweight='bold')
                                
                                # Annotation pour la série lissée
                                smooth_slice = fitted_model.fittedvalues[year_mask]
                                if len(smooth_slice) > 0:
                                    last_smooth_value = float(smooth_slice.iloc[-1])
                                    last_smooth_date = smooth_slice.index[-1]
                                    ax.annotate(f'{last_smooth_value:,.0f}',
                                               xy=(last_smooth_date, last_smooth_value),
                                               xytext=(0, -25), 
                                               textcoords='offset points',
                                               ha='center', 
                                               fontsize=9, 
                                               color='#ff0066',
                                               bbox=dict(boxstyle='round,pad=0.3', 
                                                       facecolor='#1a1f3a', 
                                                       edgecolor='#ff0066',
                                                       alpha=0.9),
                                               fontweight='bold')
                            except Exception:
                                continue
                
                # Formatter l'axe Y
                def format_y_axis(value, pos):
                    if abs(value) >= 1e12:
                        return f'{value/1e12:.1f}T'
                    elif abs(value) >= 1e9:
                        return f'{value/1e9:.1f}B'
                    elif abs(value) >= 1e6:
                        return f'{value/1e6:.1f}M'
                    else:
                        return f'{value:,.0f}'
                
                ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
                
                ax.set_title("Comparaison : Série Originale vs Lissage de Holt", 
                            color='white', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel("Date", color='#a0aec0', fontsize=12)
                ax.set_ylabel("Valeur", color='#a0aec0', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.3, color='#4dd4ff')
                ax.legend(loc='upper left', facecolor='#1a1f3a', edgecolor='#00d9ff', 
                         labelcolor='white', fontsize=11)
                ax.tick_params(colors='#a0aec0')
                for spine in ax.spines.values():
                    spine.set_color('#4dd4ff')
                    spine.set_linewidth(2)
                plt.tight_layout()
                st.pyplot(fig)
        
        # TAB 4: VALIDATION
        with tab4:
            st.markdown("## 🔍 VALIDATION DU MODÈLE")
            
            residus = fitted_model.resid
            
            # Graphique des résidus
            st.markdown("### 📉 Résidus du modèle")
            fig, ax = plt.subplots(figsize=(14, 5), facecolor='#0a0e27')
            ax.set_facecolor('#1a1f3a')
            ax.plot(residus.index, residus, color='#ff0066', marker='.', linestyle='-', linewidth=1.5)
            ax.axhline(0, color='#00d9ff', linestyle='--', linewidth=2, alpha=0.7)
            
            # Ajouter des annotations pour certains points (max et min)
            max_resid_idx = residus.abs().idxmax()
            max_resid_val = residus[max_resid_idx]
            ax.annotate(f'Max: {max_resid_val:,.0f}',
                       xy=(max_resid_idx, max_resid_val),
                       xytext=(10, 10 if max_resid_val > 0 else -20),
                       textcoords='offset points',
                       ha='left',
                       fontsize=9,
                       color='#00ffaa',
                       bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='#1a1f3a',
                               edgecolor='#00ffaa',
                               alpha=0.9),
                       fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='#00ffaa', lw=1.5))
            
            ax.set_title("Résidus du modèle de lissage exponentiel", 
                        color='white', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Date", color='#a0aec0')
            ax.set_ylabel("Résidu", color='#a0aec0')
            ax.grid(True, alpha=0.3, color='#4dd4ff')
            ax.tick_params(colors='#a0aec0')
            for spine in ax.spines.values():
                spine.set_color('#4dd4ff')
                spine.set_linewidth(2)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Test de Ljung-Box
                st.markdown("### 🧪 Test de Ljung-Box (Bruit Blanc)")
                residus_clean = residus.dropna()
                lb_test = acorr_ljungbox(residus_clean, lags=[20], return_df=True)
                
                # Formater le tableau
                lb_display = lb_test.copy()
                lb_display.columns = ['Statistique LB', 'P-value']
                lb_display['Statistique LB'] = lb_display['Statistique LB'].apply(lambda x: f"{x:.4f}")
                lb_display['P-value'] = lb_display['P-value'].apply(lambda x: f"{x:.6f}")
                
                st.dataframe(lb_display, use_container_width=True)
                
                p_value_lb = lb_test['lb_pvalue'].values[0]
                if p_value_lb > 0.05:
                    st.success(f"✅ P-value = {p_value_lb:.6f} > 0.05 : La série est un bruit blanc")
                else:
                    st.warning(f"⚠️ P-value = {p_value_lb:.6f} < 0.05 : La série n'est pas un bruit blanc")
            
            with col2:
                # Test de normalité
                st.markdown("### 🧪 Test de Shapiro-Wilk (Normalité)")
                if len(residus_clean) < 5000:
                    stat, p_value_shapiro = shapiro(residus_clean)
                    
                    result_df = pd.DataFrame({
                        'Métrique': ['Statistique', 'P-value'],
                        'Valeur': [f"{stat:.6f}", f"{p_value_shapiro:.6f}"]
                    })
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    
                    if p_value_shapiro > 0.05:
                        st.success(f"✅ P-value = {p_value_shapiro:.6f} > 0.05 : Distribution normale")
                    else:
                        st.warning(f"⚠️ P-value = {p_value_shapiro:.6f} < 0.05 : Distribution non normale")
            
            st.markdown("---")
            
            # Distribution des résidus
            st.markdown("### 📊 Distribution des résidus")
            fig, ax = plt.subplots(figsize=(14, 5), facecolor='#0a0e27')
            ax.set_facecolor('#1a1f3a')
            n, bins, patches = ax.hist(residus.dropna(), bins=20, color='#00d9ff', 
                                        edgecolor='white', alpha=0.7, density=True)
            mean_resid = residus.mean()
            ax.axvline(mean_resid, color='#ff0066', linestyle='--', linewidth=2, 
                      label=f'Moyenne = {format_number(mean_resid)}')
            ax.set_title("Distribution des résidus", color='white', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Valeur des résidus", color='#a0aec0')
            ax.set_ylabel("Densité", color='#a0aec0')
            ax.legend(facecolor='#1a1f3a', edgecolor='#00d9ff', labelcolor='white')
            ax.grid(True, alpha=0.3, color='#4dd4ff')
            ax.tick_params(colors='#a0aec0')
            for spine in ax.spines.values():
                spine.set_color('#4dd4ff')
                spine.set_linewidth(2)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Métrique moyenne des résidus
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Moyenne des résidus", format_number(residus.mean()))
            with col2:
                st.metric("📊 Écart-type des résidus", format_number(residus.std()))
            with col3:
                st.metric("📊 Max |Résidu|", format_number(abs(residus).max()))
        
        # TAB 5: PRÉVISION
        with tab5:
            st.markdown("## 🔮 PRÉVISIONS")
            
            with st.spinner("🔮 Calcul des prévisions..."):
                # Calculer les prévisions
                forecast = fitted_model.forecast(steps=n_forecast)
                
                # Intervalle de confiance
                residuals = fitted_model.resid
                sigma = np.sqrt(np.mean(residuals**2))
                h = np.arange(1, n_forecast + 1)
                se_forecast = sigma * np.sqrt(h)
                z = norm.ppf(0.975)
                lower = forecast - z * se_forecast
                upper = forecast + z * se_forecast
                
                # Afficher les prévisions
                st.markdown("### 📋 Tableau des prévisions")
                forecast_df = pd.DataFrame({
                    'Date': forecast.index.strftime('%Y-%m-%d'),
                    'Prévision': forecast.values,
                    'IC Inférieur (95%)': lower.values,
                    'IC Supérieur (95%)': upper.values
                })
                
                # Formater les valeurs numériques
                for col in ['Prévision', 'IC Inférieur (95%)', 'IC Supérieur (95%)']:
                    forecast_df[col] = forecast_df[col].apply(lambda x: format_number(x))
                
                st.dataframe(forecast_df, use_container_width=True, hide_index=True, height=250)
                
                st.markdown("---")
                
                # Graphique des prévisions
                st.markdown("### 📈 Visualisation des prévisions")
                fig, ax = plt.subplots(figsize=(14, 6), facecolor='#0a0e27')
                ax.set_facecolor('#1a1f3a')
                
                # Série historique
                ax.plot(series.index, series, color='#00d9ff', linewidth=2.5, marker='o', 
                        markersize=5, label='Données historiques', alpha=0.8)
                
                # Prévisions
                ax.plot(forecast.index, forecast, color='#00ffaa', linewidth=3, marker='s', 
                        markersize=7, label=f'Prévisions ({n_forecast} périodes)', 
                        linestyle='--', alpha=0.9)
                
                # Intervalle de confiance
                ax.fill_between(forecast.index, lower, upper, color='#ff0066', 
                               alpha=0.2, label='Intervalle de confiance 95%')
                
                # Ligne de séparation
                ax.axvline(x=series.index[-1], color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.text(series.index[-1], plt.ylim()[1]*0.95, 'Prévisions →',
                       ha='right', va='top', fontsize=11, color='gray', fontweight='bold')
                
                # Ajouter des annotations pour les prévisions
                for i, (date, value) in enumerate(zip(forecast.index, forecast)):
                    ax.annotate(f'{value:,.0f}',
                               xy=(date, value),
                               xytext=(0, 10),
                               textcoords='offset points',
                               ha='center',
                               fontsize=9,
                               color='#00ffaa',
                               bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='#1a1f3a',
                                       edgecolor='#00ffaa',
                                       alpha=0.9),
                               fontweight='bold')
                
                # Ajouter annotation pour la dernière valeur historique
                last_hist_value = series.iloc[-1]
                last_hist_date = series.index[-1]
                ax.annotate(f'Dernière valeur\n{last_hist_value:,.0f}',
                           xy=(last_hist_date, last_hist_value),
                           xytext=(-30, 20),
                           textcoords='offset points',
                           ha='right',
                           fontsize=9,
                           color='#00d9ff',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='#1a1f3a',
                                   edgecolor='#00d9ff',
                                   alpha=0.9),
                           fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='#00d9ff', lw=1.5))
                
                # Formatter l'axe Y
                def format_y_axis(value, pos):
                    if abs(value) >= 1e12:
                        return f'{value/1e12:.1f}T'
                    elif abs(value) >= 1e9:
                        return f'{value/1e9:.1f}B'
                    elif abs(value) >= 1e6:
                        return f'{value/1e6:.1f}M'
                    else:
                        return f'{value:,.0f}'
                
                ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
                
                ax.set_title("Prévisions avec Intervalle de Confiance 95%", 
                            color='white', fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel("Date", color='#a0aec0', fontsize=12)
                ax.set_ylabel("Valeur", color='#a0aec0', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.3, color='#4dd4ff')
                ax.legend(loc='upper left', facecolor='#1a1f3a', edgecolor='#00d9ff', 
                         labelcolor='white', fontsize=11)
                ax.tick_params(colors='#a0aec0')
                for spine in ax.spines.values():
                    spine.set_color('#4dd4ff')
                    spine.set_linewidth(2)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("---")
                
                # Métriques des prévisions
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Première prévision", format_number(forecast.iloc[0]))
                with col2:
                    st.metric("🎯 Dernière prévision", format_number(forecast.iloc[-1]))
                with col3:
                    growth = ((forecast.iloc[-1] - series.iloc[-1]) / series.iloc[-1]) * 100
                    st.metric("📈 Croissance prévue", f"{growth:.2f}%")
                with col4:
                    avg_forecast = forecast.mean()
                    st.metric("📊 Moyenne prévisions", format_number(avg_forecast))

else:
    # Message d'accueil
    st.markdown("""
    <div style='text-align: center; padding: 50px; background: rgba(26, 31, 58, 0.5); 
                border-radius: 20px; border: 2px solid rgba(0, 217, 255, 0.3);'>
        <h2 style='color: #00d9ff;'>👋 Bienvenue dans l'application d'analyse de séries temporelles</h2>
        <p style='color: #a0aec0; font-size: 18px; margin-top: 20px;'>
            Utilisez le panneau latéral pour charger vos données et commencer l'analyse
        </p>
        <br>
        <div style='display: flex; justify-content: center; gap: 30px; margin-top: 30px;'>
            <div style='background: rgba(0, 217, 255, 0.1); padding: 20px; border-radius: 10px; 
                        border: 1px solid rgba(0, 217, 255, 0.3); width: 200px;'>
                <h3 style='color: #00d9ff; margin-bottom: 10px;'>📁 Étape 1</h3>
                <p style='color: #a0aec0; font-size: 14px;'>Chargez votre fichier CSV ou Excel</p>
            </div>
            <div style='background: rgba(0, 255, 170, 0.1); padding: 20px; border-radius: 10px; 
                        border: 1px solid rgba(0, 255, 170, 0.3); width: 200px;'>
                <h3 style='color: #00ffaa; margin-bottom: 10px;'>⚙️ Étape 2</h3>
                <p style='color: #a0aec0; font-size: 14px;'>Configurez les paramètres d'analyse</p>
            </div>
            <div style='background: rgba(255, 0, 102, 0.1); padding: 20px; border-radius: 10px; 
                        border: 1px solid rgba(255, 0, 102, 0.3); width: 200px;'>
                <h3 style='color: #ff0066; margin-bottom: 10px;'>🚀 Étape 3</h3>
                <p style='color: #a0aec0; font-size: 14px;'>Lancez l'analyse et explorez les résultats</p>
            </div>
        </div>
        <br><br>
        <div style='background: rgba(0, 217, 255, 0.05); padding: 25px; border-radius: 15px; 
                    border-left: 4px solid #00d9ff; margin-top: 40px; text-align: left;'>
            <h3 style='color: #00d9ff; margin-bottom: 15px;'>📚 Fonctionnalités de l'application :</h3>
            <ul style='color: #a0aec0; font-size: 16px; line-height: 2;'>
                <li>✅ <strong style='color: #00ffaa;'>Exploration des données</strong> : Visualisation complète de votre série temporelle</li>
                <li>✅ <strong style='color: #00ffaa;'>Analyses statistiques</strong> : Indices de tendance, dispersion, forme et dépendance</li>
                <li>✅ <strong style='color: #00ffaa;'>Lissage Holt-Winters</strong> : Modélisation avec lissage exponentiel</li>
                <li>✅ <strong style='color: #00ffaa;'>Validation du modèle</strong> : Tests statistiques (Ljung-Box, Shapiro-Wilk)</li>
                <li>✅ <strong style='color: #00ffaa;'>Prévisions</strong> : Projections futures avec intervalles de confiance</li>
            </ul>
        </div>
        <br>
        <div style='margin-top: 30px; padding: 20px; background: rgba(255, 0, 102, 0.05); 
                    border-radius: 10px; border: 1px solid rgba(255, 0, 102, 0.3);'>
            <h4 style='color: #ff0066;'>💡 Format des données attendu :</h4>
            <p style='color: #a0aec0; font-size: 14px; margin-top: 10px;'>
                • Une colonne contenant les dates ou années<br>
                • Une colonne contenant les valeurs numériques à analyser<br>
                • Format accepté : CSV, XLSX, XLS
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Exemple de données
    st.markdown("---")
    st.markdown("### 📊 Exemple de structure de données attendue")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Format acceptable")
        example_data = pd.DataFrame({
            'DATE': [1960, 1961, 1962, 1963, 1964],
            'GDP': [541988586207, 561940310345, 603639413793, 637058551724, 684144620690]
        })
        st.dataframe(example_data, use_container_width=True)
    
    with col2:
        st.markdown("#### 📝 Description")
        st.markdown("""
        <div style='background: rgba(26, 31, 58, 0.5); padding: 20px; border-radius: 10px; 
                    border: 1px solid rgba(0, 217, 255, 0.3); height: 180px;'>
            <p style='color: #a0aec0; font-size: 14px; line-height: 1.8;'>
                <strong style='color: #00d9ff;'>• DATE :</strong> Colonne temporelle (années, dates)<br>
                <strong style='color: #00d9ff;'>• GDP :</strong> Valeurs numériques à analyser<br><br>
                <em style='color: #00ffaa;'>Les valeurs peuvent contenir des espaces ou caractères spéciaux, 
                l'application les nettoiera automatiquement.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px; margin-top: 50px;'>
    <p style='color: #4dd4ff; font-size: 14px;'>
        🚀 Développé avec Streamlit | 📊 Analyse de Séries Temporelles | 🔬 Méthode Holt-Winters
    </p>
    <p style='color: #a0aec0; font-size: 12px; margin-top: 10px;'>
        Powered by Python, Pandas, Statsmodels & Matplotlib
    </p>
</div>
""", unsafe_allow_html=True)