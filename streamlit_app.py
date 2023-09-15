import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Load your food detection model
model = tf.keras.models.load_model('./models')

class_names = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']  # Your list of class names

# Streamlit UI
st.title('Food Detection App')
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

def load_and_prep_image(uploaded_file, img_shape=224):
    """
    Reads in an image from an UploadedFile, turns it into a tensor and reshapes into
    (224, 224, 3).

    Parameters
    ----------
    uploaded_file (UploadedFile): Streamlit UploadedFile object
    img_shape (int): size to resize target image to, default 224
    """
    if uploaded_file is None:
        return None

    # Read image from the UploadedFile
    img = Image.open(uploaded_file)
    # Resize the image
    img = img.resize((img_shape, img_shape))
    # Convert to numpy array
    

    # Convert to TensorFlow tensor
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    return img_tensor

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', width= 300)

    # Perform inference on the uploaded image
    preprocessed_image = load_and_prep_image(uploaded_image)
    
    if preprocessed_image is not None:
        preprocessed_image = tf.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
        prediction = model.predict(preprocessed_image)
        pred_class = class_names[np.argmax(prediction)]

        # Display the prediction results
        st.markdown(f"<h2 style='color: green;'>Prediction: {pred_class}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: green;'>Probability: {100 * prediction.max():.2f}%</h3>", unsafe_allow_html=True)
