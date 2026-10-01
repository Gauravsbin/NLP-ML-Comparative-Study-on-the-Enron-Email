import streamlit as st
import imaplib
import email
from email.header import decode_header
import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import spacy

# Initialize spaCy and setup email account details
nlp = spacy.load("en_core_web_sm")
IMAP_SERVER = "imap.gmail.com"
EMAIL_USER = "YOUR_EMAIL"
EMAIL_PASS = "Your_app_password"


# Sample dataset for fitting the TF-IDF vectorizer and classifier
sample_texts = [
    "Your account has been suspended. Click here to verify your information.",
    "Important security update: Update your password immediately.",
    "Thank you for your recent purchase! Your order is being processed.",
    "Free gift card available. Claim now!",
    "Your invoice for the recent transaction is attached."
]
sample_labels = [1, 1, 0, 1, 0]  # 1 for phishing, 0 for safe

# Text preprocessing function
def clean_email_content(content):
    cleaned_text = BeautifulSoup(content, "html.parser").get_text()
    cleaned_text = re.sub(r'http\S+|www\S+|@\S+', '', cleaned_text)
    cleaned_text = re.sub(r'\W+', ' ', cleaned_text).lower()
    doc = nlp(cleaned_text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Classification model setup
def load_model():
    # Initialize and fit TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_vectorizer.fit(sample_texts)  # Fit on sample data
    
    # Transform sample data using the fitted TF-IDF vectorizer
    tfidf_features = tfidf_vectorizer.transform(sample_texts).toarray()

    # Initialize and fit the TF-IDF model on the sample data
    tfidf_model = LogisticRegression(max_iter=1000)
    tfidf_model.fit(tfidf_features, sample_labels)

    # Set up BERT model and tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # Prepare BERT embeddings for sample data
    bert_features = []
    for text in sample_texts:
        input_ids = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True).input_ids
        with torch.no_grad():
            bert_embedding = bert_model(input_ids)[0].mean(dim=1).numpy()
        bert_features.append(bert_embedding)
    bert_features = np.vstack(bert_features)

    # Combine TF-IDF and BERT features for sample data
    combined_features = np.hstack((tfidf_features, bert_features))

    # Set up and fit the Voting Classifier
    voting_classifier = VotingClassifier(estimators=[
        ('tfidf_model', tfidf_model),
        ('bert_model', LogisticRegression(max_iter=1000))
    ], voting='soft')
    voting_classifier.fit(combined_features, sample_labels)

    return tfidf_vectorizer, bert_tokenizer, bert_model, voting_classifier

tfidf_vectorizer, bert_tokenizer, bert_model, voting_classifier = load_model()

# Email fetching and classification function using imaplib
def fetch_and_classify_emails(num_emails=5):
    # Connect to the IMAP server
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("inbox")

    # Search and fetch the latest emails
    status, messages = mail.search(None, "ALL")
    email_ids = messages[0].split()
    emails, classifications = [], []

    for email_id in email_ids[-num_emails:]:
        # Fetch each email by ID
        status, msg_data = mail.fetch(email_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                # Parse the email message
                msg = email.message_from_bytes(response_part[1])
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8")
                
                # Extract email content
                email_content = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            email_content += part.get_payload(decode=True).decode()
                else:
                    email_content = msg.get_payload(decode=True).decode()
                
                emails.append((subject, email_content))

                # Now classify the email content
                processed_content = clean_email_content(email_content)
                tfidf_features = tfidf_vectorizer.transform([processed_content]).toarray()
                bert_features = bert_model(bert_tokenizer([processed_content], return_tensors="pt", truncation=True, padding=True).input_ids)[0].mean(dim=1).detach().numpy()
                combined_features = np.hstack((tfidf_features, bert_features))
                
                # Predict and store the classification
                label = voting_classifier.predict(combined_features)
                classifications.append("Phishing" if label == 1 else "Safe")

    mail.logout()  # Close the IMAP connection
    return pd.DataFrame([(subject, content, classification) for (subject, content), classification in zip(emails, classifications)], columns=["Subject", "Content", "Classification"])

# Streamlit UI
st.title("Standalone Email Spam Detection")
st.write("Fetch and classify recent emails without needing an API.")

if st.button("Fetch and Classify Emails"):
    with st.spinner("Classifying emails..."):
        df = fetch_and_classify_emails(num_emails=10)
        st.write("Classification Results:")
        st.dataframe(df)

        # Download button for results as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="classified_emails.csv",
            mime="text/csv"
        )
