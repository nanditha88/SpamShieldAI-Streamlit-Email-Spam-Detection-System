# app.py - Main Streamlit application
import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Email Spam Detection System",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E0F7FA, #B3E5FC);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        border-bottom: 2px solid #0D47A1;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .spam {
        color: #D32F2F;
        font-weight: bold;
    }
    .ham {
        color: #388E3C;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #2196F3;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover {
        background-color: #0D47A1;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📧 Email Spam Detection System</h1>', unsafe_allow_html=True)
st.markdown("""
This application detects whether an email is **spam** or **ham (legitimate)** using machine learning.
You can either use our pre-trained model or train a new model with your own data.
""")

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Preprocessing functions
def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_sample_data():
    """Load sample email dataset"""
    # Create a sample dataset if no data is provided
    sample_emails = [
        ("Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize.", "spam"),
        ("Hi John, just checking in to see if you're available for a meeting tomorrow at 2 PM.", "ham"),
        ("URGENT: Your bank account has been compromised. Click this link to secure your account.", "spam"),
        ("Meeting reminder: Team sync-up at 3 PM in conference room B. Please bring your reports.", "ham"),
        ("Earn $5000 per week working from home. No experience needed. Start today!", "spam"),
        ("Your Amazon order #12345 has been shipped. Track your package here.", "ham"),
        ("Free trial! Get Netflix premium for 6 months. Limited time offer.", "spam"),
        ("Project deadline extended to Friday. Please submit your work by EOD Thursday.", "ham"),
        ("You have 1 new voicemail. Call 555-1234 to listen to your message.", "spam"),
        ("The quarterly report is attached. Please review and provide feedback by Friday.", "ham"),
        ("Lose 10 pounds in 7 days with our revolutionary new diet pill!", "spam"),
        ("Reminder: Dentist appointment tomorrow at 10 AM. Please arrive 10 minutes early.", "ham"),
        ("Congratulations! You're selected for a free cruise to the Bahamas!", "spam"),
        ("Hi team, the server maintenance is scheduled for Saturday at midnight.", "ham"),
        ("Your PayPal account has been limited. Verify your identity now to restore access.", "spam"),
        ("The weekly newsletter is now available. Check out the latest updates.", "ham"),
        ("Make $10,000 in 24 hours with our proven system. Guaranteed results!", "spam"),
        ("Your invoice #INV-789 is ready. Please make payment by the due date.", "ham"),
        ("You've been pre-approved for a $50,000 loan with 0% interest!", "spam"),
        ("The client meeting has been rescheduled to next Wednesday at 11 AM.", "ham")
    ]
    
    df = pd.DataFrame(sample_emails, columns=['text', 'label'])
    return df

def train_model(df, model_type='naive_bayes'):
    """Train a spam detection model"""
    # Preprocess the text
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train selected model
    if model_type == 'naive_bayes':
        model = MultinomialNB()
    elif model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'svm':
        model = SVC(kernel='linear', probability=True)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = MultinomialNB()
    
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    
    # Store results
    results = {
        'model': model,
        'vectorizer': vectorizer,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_test': X_test
    }
    
    return results

def predict_email(model, vectorizer, email_text):
    """Predict if an email is spam or ham"""
    # Preprocess the text
    cleaned_text = preprocess_text(email_text)
    
    # Transform using TF-IDF
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    return prediction, probability

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Train Model", "Test Email", "Model Analysis", "About"])

# Home page
if page == "Home":
    st.markdown('<h2 class="sub-header">Welcome to the Email Spam Detection System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### How it works:
        1. **Train Model**: Upload your email dataset or use our sample data to train a spam detection model
        2. **Test Email**: Enter an email text to check if it's spam or ham
        3. **Model Analysis**: View performance metrics and visualizations of your trained model
        
        ### Features:
        - Support for multiple machine learning algorithms
        - Interactive visualization of results
        - Real-time email classification
        - Performance metrics and analysis
        - Word clouds for spam vs ham emails
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3067/3067256.png", width=200)
        st.markdown("""
        <div class="metric-card">
            <h3>Sample Stats</h3>
            <p>Sample dataset: 20 emails</p>
            <p>Spam: 10 emails</p>
            <p>Ham: 10 emails</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick test section
    st.markdown('<h3 class="sub-header">Quick Test</h3>', unsafe_allow_html=True)
    test_email = st.text_area("Enter an email to test (sample model will be used):", 
                              "Congratulations! You've won a free iPhone. Click here to claim your prize.")
    
    if st.button("Check if Spam"):
        if not st.session_state.trained:
            # Train a quick model with sample data
            with st.spinner("Training sample model..."):
                sample_df = load_sample_data()
                results = train_model(sample_df, 'naive_bayes')
                st.session_state.model = results['model']
                st.session_state.vectorizer = results['vectorizer']
                st.session_state.trained = True
        
        prediction, probability = predict_email(st.session_state.model, st.session_state.vectorizer, test_email)
        
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 'spam':
                st.error(f"Prediction: **SPAM** 🚨")
                st.metric("Spam Probability", f"{probability[1]*100:.2f}%")
            else:
                st.success(f"Prediction: **HAM** ✓")
                st.metric("Ham Probability", f"{probability[0]*100:.2f}%")
        
        with col2:
            # Show probabilities
            prob_df = pd.DataFrame({
                'Class': ['Ham', 'Spam'],
                'Probability': [probability[0]*100, probability[1]*100]
            })
            st.bar_chart(prob_df.set_index('Class'))

# Train Model page
elif page == "Train Model":
    st.markdown('<h2 class="sub-header">Train a New Model</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Data input options
        data_option = st.radio("Choose data source:", 
                               ["Use Sample Data", "Upload CSV File"])
        
        if data_option == "Upload CSV File":
            uploaded_file = st.file_uploader("Upload your email dataset (CSV with 'text' and 'label' columns)", type=['csv'])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"Data loaded successfully! {df.shape[0]} rows, {df.shape[1]} columns")
            else:
                df = load_sample_data()
                st.info("Using sample dataset. Upload a CSV file with 'text' and 'label' columns for custom data.")
        else:
            df = load_sample_data()
            st.info("Using sample dataset.")
    
    with col2:
        # Model selection
        model_type = st.selectbox(
            "Select Model:",
            ["Naive Bayes", "Logistic Regression", "SVM", "Random Forest"]
        )
        
        model_map = {
            "Naive Bayes": "naive_bayes",
            "Logistic Regression": "logistic_regression",
            "SVM": "svm",
            "Random Forest": "random_forest"
        }
    
    # Show dataset preview
    if st.checkbox("Show Dataset Preview"):
        st.dataframe(df.head(10))
        
        # Show dataset stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Emails", df.shape[0])
        with col2:
            spam_count = df[df['label'] == 'spam'].shape[0]
            st.metric("Spam Emails", spam_count)
        with col3:
            ham_count = df[df['label'] == 'ham'].shape[0]
            st.metric("Ham Emails", ham_count)
    
    # Train button
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            results = train_model(df, model_map[model_type])
            
            # Store in session state
            st.session_state.model = results['model']
            st.session_state.vectorizer = results['vectorizer']
            st.session_state.trained = True
            st.session_state.results = results
            st.session_state.data = df
            
            st.success("Model trained successfully!")
            
            # Display metrics
            st.markdown('<h3 class="sub-header">Model Performance</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
            with col2:
                st.metric("Precision", f"{results['precision']*100:.2f}%")
            with col3:
                st.metric("Recall", f"{results['recall']*100:.2f}%")
            with col4:
                st.metric("F1 Score", f"{results['f1']*100:.2f}%")

# Test Email page
elif page == "Test Email":
    st.markdown('<h2 class="sub-header">Test Email Classification</h2>', unsafe_allow_html=True)
    
    if not st.session_state.trained:
        st.warning("No model trained yet. Please train a model first or use the sample model.")
        if st.button("Use Sample Model"):
            with st.spinner("Loading sample model..."):
                sample_df = load_sample_data()
                results = train_model(sample_df, 'naive_bayes')
                st.session_state.model = results['model']
                st.session_state.vectorizer = results['vectorizer']
                st.session_state.trained = True
                st.success("Sample model loaded successfully!")
    
    if st.session_state.trained:
        # Email input
        email_input = st.text_area(
            "Enter the email text to classify:",
            height=200,
            value="Congratulations! You've been selected for a free vacation to Hawaii. This is a limited time offer. Click the link below to claim your prize now!"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Classify Email", type="primary"):
                with st.spinner("Analyzing email..."):
                    prediction, probability = predict_email(st.session_state.model, st.session_state.vectorizer, email_input)
                    
                    # Display result
                    st.markdown("### Classification Result")
                    
                    if prediction == 'spam':
                        st.error(f"🚨 **This email is classified as SPAM**")
                        st.metric("Spam Probability", f"{probability[1]*100:.2f}%")
                        
                        # Show warning signs
                        with st.expander("Why was this classified as spam?"):
                            st.markdown("""
                            Common spam indicators:
                            - Promises of free gifts/prizes
                            - Urgent call to action
                            - Requests for personal information
                            - Too-good-to-be-true offers
                            - Suspicious links
                            """)
                    else:
                        st.success(f"✓ **This email is classified as HAM (legitimate)**")
                        st.metric("Ham Probability", f"{probability[0]*100:.2f}%")
        
        with col2:
            # Sample emails for quick testing
            st.markdown("### Try Sample Emails")
            sample_options = {
                "Spam Example": "URGENT: Your bank account has been locked. Verify your identity immediately by clicking here.",
                "Ham Example": "Hi team, the meeting is scheduled for 3 PM today in the main conference room.",
                "Another Spam": "Earn $5000 weekly from home! No experience needed. Start today!",
                "Another Ham": "Your Amazon order has been shipped. Track your package using this link."
            }
            
            selected_sample = st.selectbox("Select a sample email:", list(sample_options.keys()))
            if st.button("Load Sample"):
                st.session_state.sample_email = sample_options[selected_sample]
                st.rerun()
            
            if 'sample_email' in st.session_state:
                st.text_area("Sample Email", st.session_state.sample_email, height=150)

# Model Analysis page
elif page == "Model Analysis":
    st.markdown('<h2 class="sub-header">Model Analysis & Visualizations</h2>', unsafe_allow_html=True)
    
    if not st.session_state.trained:
        st.warning("No model trained yet. Please train a model first.")
    else:
        results = st.session_state.results
        df = st.session_state.data
        
        # Metrics
        st.markdown("### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
        with col2:
            st.metric("Precision", f"{results['precision']*100:.2f}%")
        with col3:
            st.metric("Recall", f"{results['recall']*100:.2f}%")
        with col4:
            st.metric("F1 Score", f"{results['f1']*100:.2f}%")
        
        # Confusion Matrix
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(results['y_test'], results['y_pred'], labels=['ham', 'spam'])
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], 
                    yticklabels=['Ham', 'Spam'],
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        
        # Classification Report
        st.markdown("### Classification Report")
        report = classification_report(results['y_test'], results['y_pred'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}"))
        
        # Word Clouds
        st.markdown("### Word Frequency Analysis")
        
        if df is not None:
            # Prepare text for word clouds
            spam_text = " ".join(df[df['label'] == 'spam']['text'].apply(preprocess_text))
            ham_text = " ".join(df[df['label'] == 'ham']['text'].apply(preprocess_text))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Spam Words")
                if spam_text.strip():
                    wordcloud = WordCloud(width=400, height=300, background_color='white', 
                                          colormap='Reds', max_words=50).generate(spam_text)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("No spam emails in dataset")
            
            with col2:
                st.markdown("#### Ham Words")
                if ham_text.strip():
                    wordcloud = WordCloud(width=400, height=300, background_color='white', 
                                          colormap='Greens', max_words=50).generate(ham_text)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("No ham emails in dataset")
        
        # Model download option
        st.markdown("### Export Model")
        st.info("You can save the trained model and vectorizer for later use.")
        
        if st.button("Save Model & Vectorizer"):
            # Save model and vectorizer
            model_data = {
                'model': st.session_state.model,
                'vectorizer': st.session_state.vectorizer
            }
            
            # Convert to bytes for download
            import io
            bytes_buffer = io.BytesIO()
            pickle.dump(model_data, bytes_buffer)
            bytes_buffer.seek(0)
            
            st.download_button(
                label="Download Model",
                data=bytes_buffer,
                file_name="spam_detection_model.pkl",
                mime="application/octet-stream"
            )

# About page
else:
    st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Email Spam Detection System
    This application uses machine learning to classify emails as spam or ham (legitimate).
    
    ### Features:
    - **Multiple Algorithms**: Choose from Naive Bayes, Logistic Regression, SVM, or Random Forest
    - **Interactive Dashboard**: User-friendly interface for training and testing
    - **Visual Analytics**: Confusion matrix, performance metrics, and word clouds
    - **Real-time Prediction**: Instant classification of email text
    
    ### How It Works:
    1. **Text Preprocessing**: Emails are cleaned by removing special characters and converting to lowercase
    2. **Feature Extraction**: TF-IDF vectorization converts text to numerical features
    3. **Model Training**: Selected algorithm learns patterns from labeled email data
    4. **Classification**: New emails are classified based on learned patterns
    
    ### Technologies Used:
    - Python
    - Scikit-learn (Machine Learning)
    - Streamlit (Web Dashboard)
    - Pandas & NumPy (Data Processing)
    - Matplotlib & Seaborn (Visualization)
    
    ### Model Performance:
    The effectiveness of spam detection depends on:
    - Quality and size of training data
    - Choice of algorithm
    - Feature extraction parameters
    - Text preprocessing techniques
    
    ### Usage Tips:
    - For best results, train with a large, balanced dataset
    - Regularly update the model with new spam patterns
    - Combine multiple features (headers, content, metadata) for better accuracy
    """)
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>Email Spam Detection System &copy; 2023 | Built with Streamlit</p>
        <p>This is a demonstration application for educational purposes.</p>
    </div>
    """, unsafe_allow_html=True)
