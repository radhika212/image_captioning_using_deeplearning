import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
class CustomLSTM(layers.LSTM):
    def __init__(self, units, **kwargs):
        super().__init__(units, **kwargs)


# Load MobileNetV2 model
mobilenet_model = MobileNetV2(weights="imagenet")
mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)

# Load your trained model
model = tf.keras.models.load_model(
    'mymodel.h5', 
    custom_objects={'CustomLSTM': CustomLSTM}
)


# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
    
# Set custom web page title
st.set_page_config(page_title="Caption Generator App", page_icon="📷")

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        color: #ff6347;
        font-size: 3em;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .description {
        font-size: 1.2em;
        color: #555;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .image-container {
        text-align: center;
        margin-top: 20px;
    }
    .caption {
        border-left: 6px solid #ff6347;
        padding: 5px 20px;
        margin-top: 20px;
        font-size: 1.3em;
        font-style: italic;
        font-family: 'Georgia', serif;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.markdown('<h1 class="title">Image Caption Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">Upload an image, and this app will generate a caption for it using a trained LSTM model.</p>', unsafe_allow_html=True)

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded image
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generated Caption")
    # Display loading spinner while processing
    with st.spinner("Generating caption..."):
        # Load image
        image = load_img(uploaded_image, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Extract features using VGG16
        image_features = mobilenet_model.predict(image, verbose=0)

        # Max caption length
        max_caption_length = 34
        
        # Define function to get word from index
        def get_word_from_index(index, tokenizer):
            return next(
                (word for word, idx in tokenizer.word_index.items() if idx == index), None
        )

        # Generate caption using the model
        def predict_caption(model, image_features, tokenizer, max_caption_length):
            caption = "startseq"
            for _ in range(max_caption_length):
                sequence = tokenizer.texts_to_sequences([caption])[0]
                sequence = pad_sequences([sequence], maxlen=max_caption_length)
                yhat = model.predict([image_features, sequence], verbose=0)
                predicted_index = np.argmax(yhat)
                predicted_word = get_word_from_index(predicted_index, tokenizer)
                caption += " " + predicted_word
                if predicted_word is None or predicted_word == "endseq":
                    break
            return caption

        # Generate caption
        generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)

        # Remove startseq and endseq
        generated_caption = generated_caption.replace("startseq", "").replace("endseq", "")

    # Display the generated caption with custom styling
    st.markdown(
        f'<div class="caption">“{generated_caption}”</div>',
        unsafe_allow_html=True
    )
