# Import required libraries
# Import required libraries
import os
import base64
import re
import numpy as np
import pandas as pd
import imaplib
import email
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from hmmlearn import hmm
from collections import Counter
import nltk
import spacy

# Download nltk punkt tokenizer
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def generate_report(classifications):
    """Generates a report of the classifications and saves it as Excel and CSV."""
    df = pd.DataFrame(classifications)
    phishing_count = df[df['classification'] == 'Phishing'].shape[0]
    not_phishing_count = df[df['classification'] == 'Not Phishing'].shape[0]
    
    print("Phishing Count:", phishing_count)
    print("Not Phishing Count:", not_phishing_count)
    print(classification_report(df['classification'], df['classification']))
    
    df.to_excel('email_classification_report.xlsx', index=False, sheet_name='Email Classifications')
    df.to_csv('email_classification_report.csv', index=False)
    print("Report saved as 'email_classification_report.xlsx'.")

def load_and_split_data(file_path='emails.csv'):
    """Loads and splits the dataset into training and testing sets."""
    data = pd.read_csv(file_path)
    messages = data['message']
    labels = data['label']
    X_train, X_test, y_train, y_test = train_test_split(messages, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def extract_features(email_texts):
    """Extracts features from email texts for ML classification."""
    features = []
    for email in email_texts:
        word_count = len(email.split())
        avg_word_length = np.mean([len(word) for word in email.split()])
        num_special_chars = len(re.findall(r'[^\w\s]', email))
        num_digits = len(re.findall(r'\d', email))
        num_uppercase_words = sum(1 for word in email.split() if word.isupper())
        num_links = len(re.findall(r'http\S+', email))
        
        features.append([
            word_count,
            avg_word_length,
            num_special_chars,
            num_digits,
            num_uppercase_words,
            num_links
        ])
    return features

def fetch_emails_imap(username, password, imap_server='imap.gmail.com', mailbox='INBOX'):
    """Fetches emails using IMAP and extracts subject and body text."""
    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(username, password)
    mail.select(mailbox)

    result, data = mail.search(None, 'ALL')
    email_ids = data[0].split()

    email_data = []
    for email_id in email_ids[:10]:  # Fetch a limited number of emails (10 for example)
        result, message_data = mail.fetch(email_id, '(RFC822)')
        raw_email = message_data[0][1]
        msg = email.message_from_bytes(raw_email)

        snippet = msg['subject'] if msg['subject'] else "No Subject"
        email_body = None

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    email_body = part.get_payload(decode=True).decode('utf-8')
                    break
        else:
            email_body = msg.get_payload(decode=True).decode('utf-8')

        if email_body:
            email_data.append({
                'snippet': snippet,
                'body': email_body
            })
            print(f"Message Snippet: {snippet}")
            print(f"Email Body (Preview): {email_body[:100]}")
    
    mail.logout()
    return email_data

def preprocess_email(email):
    """Preprocesses the email text for analysis."""
    email = re.sub(r'http\S+', '', email)
    email = re.sub(r'\W', ' ', email)
    email = re.sub(r'\d', '', email)
    tokens = word_tokenize(email.lower())
    return ' '.join(tokens)

def create_and_train_ensemble_models(X_train, y_train):
    """Creates and trains ensemble models."""
    rf_model = RandomForestClassifier(n_estimators=100)
    nb_model = MultinomialNB()
    svm_model = SVC(probability=True)
    memm_model = LogisticRegression(class_weight='balanced', C=0.5)
    hmm_model = hmm.GaussianHMM(n_components=2)

    hmm_model.startprob_ = np.array([0.5, 0.5])
    hmm_model.transmat_ = np.array([[0.7, 0.3], [0.3, 0.7]])
    hmm_model.means_ = np.array([[0.0], [1.0]])
    hmm_model.covars_ = np.array([[1.0], [1.0]])
    hmm_model.init_params = ''

    tokenized_emails = [email.split() for email in X_train]
    hmm_features = np.array([len(token) for email in tokenized_emails for token in email]).reshape(-1, 1)
    lengths = [len(email) for email in tokenized_emails]

    hmm_model.fit(hmm_features, lengths)
    memm_model.fit(extract_features(X_train), y_train)

    ensemble = VotingClassifier(
        estimators=[('rf', rf_model), ('nb', nb_model), ('svm', svm_model)],
        voting='soft'
    )

    X_train_features = extract_features(X_train)
    ensemble.fit(X_train_features, y_train)

    return ensemble, memm_model, hmm_model

def classify_emails(emails, ensemble_model, memm_model, hmm_model, bert_tokenizer, bert_model):
    """Classifies emails using ensemble models, MEMM, HMM, and BERT."""
    classifications = []
    for email in emails:
        processed_email = preprocess_email(email['body'])
        
        inputs = bert_tokenizer(processed_email, return_tensors="pt", padding=True, truncation=True)
        outputs = bert_model(**inputs)
        bert_classification = int(outputs.logits.argmax().item())
        
        features = extract_features([processed_email])[0]
        ensemble_classification = int(ensemble_model.predict([features])[0])
        
        hmm_features = np.array([len(processed_email.split())]).reshape(-1, 1)
        hmm_classification = int(hmm_model.predict(hmm_features)[0])
        
        memm_classification = int(memm_model.predict([features])[0])
        
        phishing_votes = sum([
            bert_classification == 1,
            ensemble_classification == 1,
            hmm_classification == 1,
            memm_classification == 1
        ])
        
        final_classification = "Phishing" if phishing_votes >= 3 else "Not Phishing"
        
        print(f"BERT: {bert_classification}, Ensemble: {ensemble_classification}, HMM: {hmm_classification}, MEMM: {memm_classification}")
        classifications.append({'email': email, 'classification': final_classification})
    
    return classifications

def main():
    # Load and split dataset
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Extract features for ensemble training and testing
    X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test)
    
    # Train ensemble model and additional models with raw email messages for HMM, MEMM, and CRF
    ensemble_model, memm_model, hmm_model = create_and_train_ensemble_models(X_train, y_train)

    # Authenticate and fetch emails using IMAP
    username = input("Enter your email: ")
    password = input("Enter your password: ")
    emails_data = fetch_emails_imap(username, password)
    
    # BERT Model setup
    bert_tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2')
    bert_model = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2')

    # Classify fetched emails
    classifications = classify_emails(emails_data, ensemble_model, memm_model, hmm_model, bert_tokenizer, bert_model)

    # Generate and save report as Excel file
    generate_report(classifications)

if __name__ == '__main__':
    main()
