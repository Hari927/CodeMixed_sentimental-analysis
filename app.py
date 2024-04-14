import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
import pickle
import requests

# Apply background and styling directly within the app
def set_bg_hack_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1522252234503-e356532cafd5");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Define the custom f1_metric function exactly as it was during training
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# Load your trained model
model_url = "https://github.com/Hari927/CodeMixed_sentimental-analysis/raw/main/sentiment%20analysis.h5"
model_response = requests.get(model_url)
with open("sentiment_analysis.h5", "wb") as model_file:
    model_file.write(model_response.content)
model = load_model("sentiment_analysis.h5", custom_objects={'f1_metric': f1_metric})

# Load your tokenizer
tokenizer_url = "https://github.com/Hari927/CodeMixed_sentimental-analysis/raw/main/tokenizer.pickle"
tokenizer_response = requests.get(tokenizer_url)
with open("tokenizer.pickle", "wb") as tokenizer_file:
    tokenizer_file.write(tokenizer_response.content)
with open("tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 80  # Adjust this value to match the input shape your model expects

# Function to preprocess the text
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return padded_sequences

# Function to predict sentiment
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text).round()
    if prediction[0][0] == 1.0:
        return 'Positive sentiment 😊'
    else:
        return 'Negative sentiment 😢'

# Streamlit app main function
def main():
    # Set background
    set_bg_hack_url()

    st.title('Sentimeix - Sentiment Analysis')
    user_input = st.text_area("Enter your text below:", "")

    if st.button('Predict Sentiment'):
        sentiment = predict_sentiment(user_input)
        st.write(f'Sentiment: {sentiment}')

    # Footer
    st.markdown("---")
    st.markdown("Thanks for visiting! ❤️ Made with love ")

if __name__ == '__main__':
    main()
