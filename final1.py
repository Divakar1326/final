import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, r2_score, classification_report,accuracy_score,f1_score,mean_squared_error
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from transliterate import translit
from indic_transliteration import sanscript
from unidecode import unidecode
import icu
from PIL import Image
import base64
st.set_page_config(
    page_title="Multilingual Translator",
    page_icon="🔠",
    layout="wide",
    initial_sidebar_state="expanded")
st.markdown(
   """
<style>
.st-emotion-cache-6qob1r.eczjsme11{
         background-color: rgba(255, 255, 255, 0.2) !important;
        border-right: 2px solid black;  
        border-radius: 5px;  
        color: black !important;
}
</style>
""",unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .header-title {
        font-size: 50px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True
)
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_background("Pictures/finalpp.jpg")
translator = GoogleTranslator()
@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, bert_model
tokenizer, bert_model = load_bert_model()
def get_sentence_embedding(sentence):
    vectorizer = TfidfVectorizer()
    sentence_embedding = vectorizer.fit_transform([sentence])
    return np.mean(sentence_embedding.toarray(), axis=0)
def extract_tfidf_features(sentences):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(sentences).toarray()
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
def generate_dummy_data(num_samples=100):
    embeddings = np.random.rand(num_samples, 768)  # Dummy sentence embeddings (768 features)
    labels = np.random.choice([0, 1, 2], size=num_samples)
    return embeddings, labels
st.title("🔠 Translation Evaluation with Multiple ML Models")
def transliterate_text(text, lang_code):
    try:
        return translit(text, lang_code, reversed=True)  # Reverse to get Romanization
    except Exception as e:
        return f"Error in transliteration: {str(e)}"
def transliterate(text, lang_code):
    try:
        transliterators = {
            'ru': icu.Transliterator.createInstance("Cyrillic..Latin"),
            'bg': icu.Transliterator.createInstance("Cyrillic..Latin"),
            'el': icu.Transliterator.createInstance("Greek..Latin"),
            'ja': icu.Transliterator.createInstance("Japanese..Latin"),
            'zh-CN': icu.Transliterator.createInstance("Han..Latin"),
            'zh-TW': icu.Transliterator.createInstance("Han..Latin"),
            'ar': icu.Transliterator.createInstance("Arabic..Latin"),
        }

        romanization_methods = {
            'hi': lambda t: sanscript.transliterate(t, sanscript.DEVANAGARI, sanscript.IAST),
            'bn': lambda t: sanscript.transliterate(t, sanscript.BENGALI, sanscript.IAST),
            'mr': lambda t: sanscript.transliterate(t, sanscript.DEVANAGARI, sanscript.IAST),
            'gu': lambda t: sanscript.transliterate(t, sanscript.GUJARATI, sanscript.IAST),
            'ta': lambda t: sanscript.transliterate(t, sanscript.TAMIL, sanscript.IAST),
            'te': lambda t: sanscript.transliterate(t, sanscript.TELUGU, sanscript.IAST),
            # Use unidecode for other languages as a fallback
            'ru': lambda t: unidecode(t),
            'bg': lambda t: unidecode(t),
            'el': lambda t: unidecode(t),
            'ja': lambda t: unidecode(t),
            'zh-CN': lambda t: unidecode(t),
            'zh-TW': lambda t: unidecode(t),
            'ar': lambda t: unidecode(t),
        }

        if lang_code in romanization_methods:
            return romanization_methods[lang_code](text)
        elif lang_code in transliterators:
            return transliterators[lang_code].transliterate(text)
        else:
            return unidecode(text)  # Default fallback
    except Exception as e:
        return f"Error in transliteration: {str(e)}"
selected_model = st.sidebar.radio("EXPLORE 🔎", ['🎖️ Naive Bayes', '🎖️ Logistic Regression', '🎖️ SVM', '🎖️ Linear Regression', '🎖️ KNN', '🎖️ K-Means'])
language_map = {
    'Afrikaans': 'af', 'Albanian': 'sq', 'Arabic': 'ar', 'Armenian': 'hy',
    'Azerbaijani': 'az', 'Basque': 'eu', 'Belarusian': 'be', 'Bengali': 'bn',
    'Bosnian': 'bs', 'Bulgarian': 'bg', 'Catalan': 'ca', 'Chinese (Simplified)': 'zh-CN',
    'Chinese (Traditional)': 'zh-TW', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da',
    'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 'Estonian': 'et', 'Filipino': 'tl',
    'Finnish': 'fi', 'French': 'fr', 'Galician': 'gl', 'Georgian': 'ka', 'German': 'de',
    'Greek': 'el', 'Gujarati': 'gu', 'Haitian Creole': 'ht', 'Hebrew': 'iw', 'Hindi': 'hi',
    'Hungarian': 'hu', 'Icelandic': 'is', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it',
    'Japanese': 'ja', 'Javanese': 'jw', 'Kannada': 'kn', 'Kazakh': 'kk', 'Khmer': 'km',
    'Korean': 'ko', 'Kurdish (Kurmanji)': 'ku', 'Lao': 'lo', 'Latin': 'la', 'Latvian': 'lv',
    'Lithuanian': 'lt', 'Macedonian': 'mk', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt',
    'Marathi': 'mr', 'Mongolian': 'mn', 'Nepali': 'ne', 'Norwegian': 'no', 'Persian': 'fa',
    'Polish': 'pl', 'Portuguese': 'pt', 'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru',
    'Serbian': 'sr', 'Sinhala': 'si', 'Slovak': 'sk', 'Slovenian': 'sl', 'Spanish': 'es',
    'Sundanese': 'su', 'Swahili': 'sw', 'Swedish': 'sv', 'Tamil': 'ta', 'Telugu': 'te',
    'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz',
    'Vietnamese': 'vi', 'Welsh': 'cy', 'Xhosa': 'xh', 'Yiddish': 'yi', 'Yoruba': 'yo',
    'Zulu': 'zu'
}
selected_language = st.sidebar.selectbox("Select Target Language: 🚀", list(language_map.keys()))
selected_language_code = language_map[selected_language]
input_sentence = st.text_input("Enter a sentence: 👇", "")
if input_sentence:
    translated_sentence = GoogleTranslator(source='auto', target=selected_language_code).translate(input_sentence)
    st.write('---')
    st.subheader(f"**🎰 Translated Sentence ({selected_language}):** {translated_sentence}")
    st.write('---')
    romanized_sentence = transliterate_text(translated_sentence, selected_language_code)
    st.subheader(f"**📈 Romanized Sentence:** {romanized_sentence}",divider='red')
X, y = generate_dummy_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_train_scaled, X_test_scaled = scale_features(X_train_smote, X_test)
if selected_model == '🎖️ Naive Bayes':
    model = GaussianNB()
    model.fit(X_train_scaled, y_train_smote)
    predictions = model.predict(X_test_scaled)  
elif selected_model == '🎖️ Logistic Regression':
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train_smote)
    predictions = model.predict(X_test_scaled)
elif selected_model == '🎖️ SVM':
    model = SVC()
    model.fit(X_train_scaled, y_train_smote)
    predictions = model.predict(X_test_scaled)
elif selected_model == '🎖️ Linear Regression':
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_smote)
    predictions = model.predict(X_test_scaled).round()
elif selected_model == '🎖️ KNN':
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train_smote)
    predictions = model.predict(X_test_scaled)  
elif selected_model == '🎖️ K-Means':
    model = KMeans(n_clusters=3)
    model.fit(X_train_scaled)
    predictions = model.predict(X_test_scaled)
st.header("**Evaluation Metrics 📝**",divider='red')
if selected_model in ['🎖️ Naive Bayes', '🎖️ Logistic Regression', '🎖️ SVM', '🎖️ KNN', '🎖️ K-Means']:
    st.write(f"<span style='color:#FFA500; font-size:30px;'>Accuracy 🎯</span>: {accuracy_score(y_test, predictions):.2f}", unsafe_allow_html=True)
    st.write('---')
    st.write(f"<span style='color:#FFA500; font-size:30px;'>F1-Score 💢</span>: {f1_score(y_test, predictions, average='weighted'):.2f}", unsafe_allow_html=True)
    report = classification_report(y_test, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write('---')
    st.write("<span style='color:#FFA500; font-size:30px;'>Classification Report (DataFrame Format) 📄</span>:", unsafe_allow_html=True)
    st.dataframe(report_df)
    conf_matrix = confusion_matrix(y_test, predictions)
    st.write('---')
    st.write("<span style='color:#FFA500; font-size:30px;'>confussion matrix (DataFrame Format) ❌</span>:", unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
elif selected_model == '🎖️ Linear Regression':
    st.write(f"<span style='color:#FFA500; font-size:30px;'>R² Score 📄</span>: {r2_score(y_test, predictions):.2f}",unsafe_allow_html=True)
    st.write('---')
    st.write(f"<span style='color:#FFA500; font-size:30px;'>Mean Squared Error (MSE) 💢</span>: {mean_squared_error(y_test, predictions):.2f}",unsafe_allow_html=True)
    st.write('---')
    st.write(f"<span style='color:#FFA500; font-size:30px;'>Mean Absolute Error (MAE) 💢</span>: {mean_absolute_error(y_test, predictions):.2f}",unsafe_allow_html=True)
    corr = np.corrcoef(y_test, predictions)[0, 1]
    st.write('---')
    st.write(f"<span style='color:#FFA500; font-size:30px;'>Correlation ❌</span>: {corr:.2f}",unsafe_allow_html=True)
    st.write('---')
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=predictions, ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
if st.checkbox("Show Heatmap of Feature Correlation"):
    correlation_matrix = pd.DataFrame(X_train_scaled).corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)
st.write('---')
def generate_dummy_data(num_samples=100):
    np.random.seed(42)
    data = pd.DataFrame({
        'Feature_1': np.random.rand(num_samples),
        'Feature_2': np.random.rand(num_samples) * 2,
        'Feature_3': np.random.rand(num_samples) * 3,
        'Target': np.random.choice([0, 1, 2], size=num_samples)
    })
    return data
st.header("Correlation Analysis with Heatmap 📝",divider='red')
data = generate_dummy_data()
st.write("<span style='color:#008000; font-size:30px;'>Dataset preview ✉️</span>:", unsafe_allow_html=True)
st.dataframe(data.head())
correlation_matrix = data.corr()
st.write('---')
st.write("<span style='color:#008000; font-size:30px;'>Correlation Matrix 📑</span>:", unsafe_allow_html=True)
st.dataframe(correlation_matrix)
st.write('---')
st.write("<span style='color:#008000; font-size:30px;'>Correlation Heatmap 🔥</span>:", unsafe_allow_html=True)
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
st.pyplot(plt)
st.write('---')
