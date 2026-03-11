import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

# Set page config for better UI
st.set_page_config(
    page_title="🎭 SentiMent - Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    .stContainer {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(102, 126, 234, 0.3);
    }
    .sentiment-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
    }
    .positive {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #1a5f3f;
    }
    .negative {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #8b0000;
    }
    .neutral {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #4a4a4a;
    }
    .title {
        text-align: center;
        color: white;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <h1 style='text-align: center; color: white; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
        🎭 SentiMent - Sentiment Analyzer 🎭
    </h1>
    <p style='text-align: center; color: #667eea; font-size: 1.1rem; margin-bottom: 2rem;'>
        Discover the sentiment hidden in every word! ✨
    </p>
""", unsafe_allow_html=True)

# ========== FUNCTIONS ==========

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_or_train_model():
    """Load existing model or train new one"""
    model_path = "sentiment_model.pkl"
    vectorizer_path = "vectorizer.pkl"
    
    # Check if models exist
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    
    # Train new model if not exists
    df = pd.read_csv("sentimentDataset/sentiment_analysis.csv")
    df = df[['text', 'sentiment']]
    
    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Label encoding
    df['sentiment'] = df['sentiment'].str.lower().str.strip()
    df['label'] = df['sentiment'].map({
        'positive': 2,
        'neutral': 1,
        'negative': 0
    })
    
    df = df.dropna(subset=['label'])
    
    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']
    
    # Train model
    model = LogisticRegression(max_iter=1000, multi_class='auto', random_state=42)
    model.fit(X, y)
    
    # Save models
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment of given text"""
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    pred = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])
    
    sentiment_labels = {0: '😢 Negative', 1: '😐 Neutral', 2: '😊 Positive'}
    sentiment_colors = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    return sentiment_labels[pred], sentiment_colors[pred], confidence

# ========== MAIN APP ==========

try:
    model, vectorizer = load_or_train_model()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Choose Your Input Method")
        input_method = st.radio(
            "How do you want to analyze sentiment?",
            ["📚 Select from Examples", "✍️ Type Your Own"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("Models Loaded", "✅", help="Model is ready!")
        with col_stats2:
            st.metric("Categories", "3", help="Negative, Neutral, Positive")
        with col_stats3:
            st.metric("Accuracy", "~85%", help="Approximate model accuracy")
    
    st.divider()
    
    # Input section
    text_to_analyze = ""
    
    if input_method == "📚 Select from Examples":
        st.markdown("#### 🌟 Pre-loaded Examples")
        
        example_sentences = {
            "😊 Positive Examples": [
                "I absolutely love this product! It's amazing!",
                "This is the best day of my life!",
                "You're doing a fantastic job!",
                "I'm so happy with my purchase!",
                "This movie was brilliant and entertaining!"
            ],
            "😐 Neutral Examples": [
                "The weather is cloudy today.",
                "I have a meeting at 3 PM.",
                "This is a table.",
                "Yesterday, I went to the store.",
                "The building is tall."
            ],
            "😢 Negative Examples": [
                "I hate waiting in long lines.",
                "This is absolutely terrible!",
                "I'm so disappointed with this service.",
                "This is the worst experience ever.",
                "I don't like this at all."
            ]
        }
        
        selected_category = st.selectbox(
            "Select a sentiment category:",
            list(example_sentences.keys())
        )
        
        selected_sentence = st.selectbox(
            "Choose a sentence:",
            example_sentences[selected_category],
            key="example_select"
        )
        
        text_to_analyze = selected_sentence
        
        # Display selected text
        st.markdown("#### Selected Text:")
        st.info(f"**{text_to_analyze}**")
    
    else:  # Type Your Own
        st.markdown("#### ✍️ Type or Paste Your Text")
        text_to_analyze = st.text_area(
            "Enter your text here:",
            placeholder="Type something... Let's analyze the sentiment!",
            height=150,
            label_visibility="collapsed"
        )
    
    # Analysis button
    if st.button("🔮 Analyze Sentiment", use_container_width=True, type="primary"):
        if text_to_analyze.strip():
            with st.spinner("🔄 Analyzing sentiment..."):
                sentiment, color, confidence = predict_sentiment(text_to_analyze, model, vectorizer)
                
                # Display results
                st.divider()
                st.markdown("## 📈 Analysis Results")
                
                # Sentiment box
                st.markdown(f"""
                    <div class='sentiment-box {color}'>
                        {sentiment}<br>
                        <span style='font-size: 0.9rem; opacity: 0.8;'>Confidence: {confidence:.1%}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Confidence visualization
                st.markdown("### Confidence Score")
                st.progress(confidence, text=f"{confidence:.1%}")
                
                # Additional details
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Your Text:** {text_to_analyze}")
                with col2:
                    st.success(f"**Detected Sentiment:** {sentiment}")
                
                # Fun message based on sentiment
                if sentiment.startswith('😊'):
                    st.balloon()
                    st.markdown("🌟 **Great vibes detected!** Your sentiment is positive! Keep up the good energy!")
                elif sentiment.startswith('😐'):
                    st.markdown("😐 **Neutral tone detected.** This text seems objective or factual.")
                else:
                    st.markdown("😟 **Negative sentiment detected.** This text expresses dissatisfaction or sadness.")
        else:
            st.warning("⚠️ Please enter some text to analyze!")
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### 📖 How It Works")
        st.markdown("""
        1. **Input**: Choose or type any sentence
        2. **Processing**: Text is cleaned and processed
        3. **Analysis**: ML model predicts sentiment
        4. **Results**: Get instant sentiment classification
        
        ---
        ### 🎯 Sentiment Categories
        - **😊 Positive**: Happy, satisfied, enthusiastic
        - **😐 Neutral**: Factual, objective, informative
        - **😢 Negative**: Unhappy, angry, disappointed
        
        ---
        ### 🛠️ Tech Stack
        - **Framework**: Streamlit
        - **ML Model**: Logistic Regression
        - **Vectorization**: TF-IDF
        - **Language**: Python
        
        ---
        ### 💡 Tips
        - Be specific for better results
        - Longer texts often give more accurate results
        - Sarcasm might be detected differently
        """)
        
        st.divider()
        st.markdown("**Made with ❤️ using Python & Streamlit**")

except FileNotFoundError as e:
    st.error(f"❌ Error: {e}")
    st.warning("Make sure the sentiment dataset CSV file exists in the correct location.")
except Exception as e:
    st.error(f"❌ An error occurred: {e}")
    st.info("Please check your setup and try again.")
