# Spam Classification System

This is a web-based spam classification system built with **Streamlit**, **Python**, and **Machine Learning**. The application uses a trained model to classify text messages as either **Spam** or **Not Spam**.

## Demo

You can try out the live demo of the Spam Classification System here:

[Spam Classification System - Live Demo](https://aryansachan12-spam-classification-app-c7wgvs.streamlit.app/)

## Features

- **Spam Detection**: Enter a message, and the app will classify it as "Spam" or "Not Spam".
- **Pre-trained Model**: The system uses a trained machine learning model to make predictions.
- **Automatic Model Training**: If the model file (`spam_classifier.pkl`) is not found, the app automatically runs a Jupyter notebook to train and save the model.

## Installation

### Prerequisites

To run the application locally, you will need to have the following libraries installed:

- Python 3.x  
- Streamlit  
- Pandas  
- Scikit-learn  
- Jupyter  
- Pickle  

### Steps to Run the Project Locally

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/spam-classification.git
    ```

2. Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

The app will be available at `http://localhost:8501` in your browser.

## How It Works

- **Dataset**: The application uses a labeled dataset of SMS messages, where each message is marked as "spam" or "ham" (not spam).
- **Text Processing**: The text is cleaned, vectorized using techniques like TF-IDF, and fed into a classification model.
- **Model Generation**: On startup, the app checks for the presence of `spam_classifier.pkl`. If not found, it runs `train_model.ipynb` to train the model and save it.
- **Prediction Logic**: The model predicts the label based on message features, and the app displays the result in real-time.

## Files Included

- `app.py`: The main Streamlit app for spam classification.
- `train_model.ipynb`: A Jupyter notebook for training and saving the spam classification model.
- `spam_classifier.pkl`: The pickled trained model file (generated after running the notebook).
- `vectorizer.pkl`: The pickled vectorizer used to transform text data (generated after running the notebook).

## Future Improvements

- Add support for multilingual spam detection.
- Display word clouds or keyword highlights.
- Log predictions to improve future model versions.
- Allow batch upload of messages (e.g., from CSV).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
