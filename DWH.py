import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Import FigureCanvasTkAgg
import seaborn as sns  # Make sure seaborn is installed properly
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Function to preprocess data
def preprocess_data(data):
    data['Sentiment'] = data['Sentiment'].map({'positive': 1, 'negative': 0, 'neutral': 2})
    return data

# Function to train model
def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    return model, vectorizer

# Function to evaluate model
def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, target_names=['negative', 'positive', 'neutral'])
    cm = confusion_matrix(y_test, y_pred)
    return report, cm

# Function to display GUI
def create_gui():
    # Load the dataset
    data = pd.read_csv("C:/Users/Desktop/Datascience Sentiment LogisticRegression and classcification/data.csv")
    
    # Preprocess the data
    data = preprocess_data(data)
    
    # Split the dataset
    X = data['Review']
    y = data['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model, vectorizer = train_model(X_train, y_train)
    
    # Evaluate the model
    report, cm = evaluate_model(model, vectorizer, X_test, y_test)
    
    # Create GUI window
    root = tk.Tk()
    root.title("Sentiment Analysis GUI")
    
    # Add labels and buttons
    label = ttk.Label(root, text="Classification Report:\n" + report, justify='left')
    label.pack(padx=10, pady=10)
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'positive', 'neutral'], yticklabels=['negative', 'positive', 'neutral'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Embedding plot in GUI
    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    # Display GUI
    root.mainloop()

# Entry point of the script
if __name__ == "__main__":
    create_gui()
