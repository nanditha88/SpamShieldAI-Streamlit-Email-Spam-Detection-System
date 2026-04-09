SpamShieldAI – Streamlit Email Spam Detection System
________________________________________

SpamShieldAI is a smart, interactive email spam detection system built using Machine Learning + Natural Language Processing (NLP) + Streamlit. The model detects whether an email message is Spam or Ham (legitimate) using ML algorithms such as Naive Bayes, Logistic Regression, SVM, and Random Forest.
The system includes real-time email classification, model training, visual analytics, and word cloud visualizations, making it ideal for learning, research, and practical use.

________________________________________

Table of Contents


•	Features 
•	How the System Works 
•	Tech Stack 
•	Project Structure 
•	Installation 
•	Usage 
•	Future Enhancements 
•	Author 
•	License 
________________________________________

Features


•	Support for multiple machine learning algorithms 
•	Interactive Streamlit dashboard 
•	Real-time email spam prediction 
•	Visual analytics: 
o	Confusion matrix 
o	Accuracy, precision, recall, F1 score 
o	Word clouds for spam and ham emails 
•	Ability to train model using sample dataset 
•	Live input box to classify custom emails 
•	Clean, fast, user-friendly UI 
________________________________________
How the System Works


Train Model
Upload your dataset (or use the included one) to train different ML models. 

Text Preprocessing 
o	Remove special characters 
o	Convert to lowercase 
o	Tokenize 
o	Stopword removal 
o	Lemmatization 

Feature Extraction
Converts email text into numerical form using TF-IDF Vectorization. 

Model Training
Selected algorithm learns from labeled emails. 

Email Classification
User enters an email → model predicts Spam / Ham instantly. 

Visual Analytics
Performance metrics + charts + word clouds. 
________________________________________

 Tech Stack
 
 
Programming & ML

•	Python 
•	Scikit-learn 
•	Pandas & NumPy 
•	Natural Language Processing (NLP) 

Visualization

•	Matplotlib 
•	Seaborn 
•	WordCloud 

Frontend (UI)

•	Streamlit 
•	HTML & CSS (Streamlit components) 

________________________________________

Project Structure


SpamShieldAI – Streamlit Email Spam Detection System/

│

├── app.py                   # Main Streamlit app

├── create_csv.py            # CSV creation script

├── emails.csv               # Dataset

├── install.bat              # Installation helper

├── requirements.txt         # Dependencies

├── setup.py                 # Setup configuration

├── README.md                # Documentation

└── .gitignore
________________________________________

Installation


•	Clone the Repository
git clone https://github.com/nanditha88/email-spam-detection.git
cd email-spam-detection

•	Create Virtual Environment
python -m venv venv

•	Activate Environment
Windows:
venv\Scripts\activate

•	Install Requirements
pip install -r requirements.txt

•	Run Streamlit App
streamlit run app.py
________________________________________

Usage


Once the web app opens in the browser:

Train Tab

•	Upload dataset or use default CSV 
•	Select algorithm 
•	Train model 
•	View accuracy & metrics 

Predict Tab

•	Paste any email text 
•	Click Predict 
•	Instantly get Spam / Ham output 

Analytics Tab

•	Confusion matrix 
•	Word clouds 
•	Accuracy chart
________________________________________

Future Enhancements


•	Improve dataset with larger, hybrid email corpus 
•	Add neural networks (LSTM / BERT) 
•	Add email file (.eml/.txt) upload feature 
•	Build API for external integration 
•	Deploy online (Streamlit Cloud / Render / AWS) 
•	Android App version in Kotlin for mobile email checks 
________________________________________

Author 


Developed by: Nanditha CL
GitHub: @nanditha88
_______________________________________   


