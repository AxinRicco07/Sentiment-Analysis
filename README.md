# Sentiment Analysis using FastAPI and ONNX
### Overview
>This project is a sentiment analysis web application that uses a pre-trained ONNX model to classify emotions from text input. The backend is built using FastAPI, and the user interface is implemented with Streamlit.
### Features
+ Predicts emotions from user input text.
+ Uses an ONNX model for efficient inference.
+ FastAPI backend for handling requests.
+ Streamlit frontend for user interaction.
+ Outputs sentiment category along with confidence scores.
### Technologies Used
**FastAPI**: For building the backend API.
**ONNX Runtime**: For running the emotion classification model.
**Transformers (Hugging Face)**: For tokenizing input text.
**Streamlit**: For creating a simple UI.
**Python**: Primary programming language.
### Installation

1. Clone the Repository
 ``````
```cd sentiment-analysis```
2. Install Dependencies
```pip install -r requirements.txt```
3. Start FastAPI Server
```uvicorn app:app --reload```

