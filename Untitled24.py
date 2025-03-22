#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[8]:


# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")


# In[9]:


# Rename columns
df.rename(columns={"v1": "label", "v2": "message"}, inplace=True)


# In[10]:



# Check dataset info
df.info()


# In[11]:



# Display first few rows
df.head()


# In[12]:


# Remove unwanted columns
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
df.info()
df.head()


# In[13]:


#preprocessing
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# In[14]:



# Download stopwords
nltk.download("stopwords")
nltk.download("punkt")


# In[15]:



# Convert labels to binary (ham → 0, spam → 1)
df["label"] = df["label"].map({"ham": 0, "spam": 1})


# In[16]:



# Text cleaning function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return " ".join(tokens)


# In[17]:



# Apply preprocessing
df["cleaned_message"] = df["message"].apply(preprocess_text)


# In[18]:



# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_message"], df["label"], test_size=0.2, random_state=42
)


# In[19]:



# Convert text data to numerical using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# In[20]:


#Training a Classification Model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# In[21]:



# Training a Naïve Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# In[22]:



# Predictions
y_pred = model.predict(X_test_tfidf)


# In[23]:



# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[24]:


#Using this Naive Bayes model we got an accuracy of 96%.
#I am trying other classification models like Support Vector Machine (SVM) and Random Forest to see which model classifies well.


# In[25]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# In[26]:


# Train and Evaluate Multiple Models
models = {
    "SVM": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)  # Train model
    y_pred = model.predict(X_test_tfidf)  # Make predictions
    print(f"\n{name} Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))


# In[27]:


# SVM model achieved highest accuracy with 98%.


# In[29]:


import joblib


# In[34]:



# Retrieve the trained SVM model from the dictionary
svm_model = models["SVM"]  

# Save the trained SVM model
joblib.dump(svm_model, "spam_sms_classifier.pkl")


# In[36]:



# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


# In[37]:


# Load the saved model and vectorizer
loaded_model = joblib.load("spam_sms_classifier.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")


# In[52]:



def predict_message(text):
    text_cleaned = preprocess_text(text)  # Preprocess input text
    text_tfidf = loaded_vectorizer.transform([text_cleaned])  # Convert to TF-IDF
    prediction = loaded_model.predict(text_tfidf)  # Predict spam or ham
    return "Spam" if prediction[0] == "spam" else "Not Spam"

# Example Predictions
message1 = "Hey, let's meet for coffee tomorrow."


# In[53]:


print(predict_message(message2))  # Expected Output: Not Spam


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




