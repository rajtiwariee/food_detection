import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_option("deprecation.showfileUploaderEncoding", False)



CLASS_NAMES = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
    "bruschetta",
    "caesar_salad",
    "cannoli",
    "caprese_salad",
    "carrot_cake",
    "ceviche",
    "cheese_plate",
    "cheesecake",
    "chicken_curry",
    "chicken_quesadilla",
    "chicken_wings",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "clam_chowder",
    "club_sandwich",
    "crab_cakes",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs_benedict",
    "escargots",
    "falafel",
    "filet_mignon",
    "fish_and_chips",
    "foie_gras",
    "french_fries",
    "french_onion_soup",
    "french_toast",
    "fried_calamari",
    "fried_rice",
    "frozen_yogurt",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "guacamole",
    "gyoza",
    "hamburger",
    "hot_and_sour_soup",
    "hot_dog",
    "huevos_rancheros",
    "hummus",
    "ice_cream",
    "lasagna",
    "lobster_bisque",
    "lobster_roll_sandwich",
    "macaroni_and_cheese",
    "macarons",
    "miso_soup",
    "mussels",
    "nachos",
    "omelette",
    "onion_rings",
    "oysters",
    "pad_thai",
    "paella",
    "pancakes",
    "panna_cotta",
    "peking_duck",
    "pho",
    "pizza",
    "pork_chop",
    "poutine",
    "prime_rib",
    "pulled_pork_sandwich",
    "ramen",
    "ravioli",
    "red_velvet_cake",
    "risotto",
    "samosa",
    "sashimi",
    "scallops",
    "seaweed_salad",
    "shrimp_and_grits",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "tuna_tartare",
    "waffles",
]


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(
        "model.h5"
    )
    return model


def load_and_prep(filename, img_shape=224):
    img = tf.cast(filename, tf.float32)
    img = tf.image.resize(img, [img_shape, img_shape])
    return img


def predict(image, model):
    image = load_and_prep(image)
    pred_prob = model.predict(
        tf.expand_dims(image, axis=0), verbose=0
    )  # make prediction on image with shape [None, 224, 224, 3]
    pred_class = CLASS_NAMES[pred_prob.argmax()]
    pred_class = pred_class.replace("_", " ").capitalize()
    prob_pred_class = tf.reduce_max(pred_prob).numpy() * 100
    prob_class_str = "{:.2f}".format(prob_pred_class)
    st.success(f"It is a **{pred_class}** with {prob_class_str}% confidence")





with col2:
    st.markdown(
        """
    ### Predict on your image!
    """
    )
    if st.checkbox("Show labels"):
        st.write("There are 101 classes, so these are some 5 labels")
        import random

        st.write(random.sample(CLASS_NAMES, 5))
    image = st.file_uploader(label="Upload an image", type=["png", "jpg", "jpeg"])
    if image is not None:
        st.image(image=image)
        test_image = Image.open(image)
        model = load_model()
        if st.button("Predict"):
            predict(test_image, model)
