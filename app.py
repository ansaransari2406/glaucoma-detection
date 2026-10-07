import os
import numpy as np
import xgboost as xgb
import joblib
import streamlit as st

from PIL import Image

from tensorflow.keras.applications import (
    ResNet50,
    DenseNet121,
    VGG16,
    InceptionV3
)

from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet_preprocess
)

from tensorflow.keras.applications.densenet import (
    preprocess_input as densenet_preprocess
)

from tensorflow.keras.applications.vgg16 import (
    preprocess_input as vgg_preprocess
)

from tensorflow.keras.applications.inception_v3 import (
    preprocess_input as inception_preprocess
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GLAUCOMA DETECTION USING DEEP LEARNING BASED PRETRAINED CNN MODELS AND ENSEMBLE LEARNING",
    page_icon="👁️",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "xgb_models"
)

# ============================================================
# TITLE
# ============================================================

st.title("👁️ Glaucoma Detection System")

st.write(
    "Deep Learning-Based Glaucoma Detection "
    "Using CNN Feature Extraction and XGBoost Ensemble Learning."
)


st.markdown("---")


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("🧠 Models Used")

st.write(
    """
    - ResNet50
    - DenseNet121
    - VGG16
    - InceptionV3
    - XGBoost
    - Logistic Regression Ensemble
    """
)


# ============================================================
# LOAD CNN MODELS
# ============================================================

@st.cache_resource
def load_cnn_models():

    models = {}

    models["resnet50"] = (
        ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3)
        ),
        resnet_preprocess
    )

    models["densenet121"] = (
        DenseNet121(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3)
        ),
        densenet_preprocess
    )

    models["vgg16"] = (
        VGG16(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3)
        ),
        vgg_preprocess
    )

    models["inceptionv3"] = (
        InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3)
        ),
        inception_preprocess
    )

    return models


# ============================================================
# LOAD XGBOOST MODELS
# ============================================================

@st.cache_resource
def load_xgb_models():

    models = {}

    model_names = [
        "resnet50",
        "densenet121",
        "vgg16",
        "inceptionv3"
    ]

    for name in model_names:

        model_path = os.path.join(
            MODEL_DIR,
            f"{name}_xgb_model.json"
        )

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"XGBoost model not found:\n{model_path}"
            )

        booster = xgb.Booster()

        booster.load_model(
            model_path
        )

        models[name] = booster

    return models


# ============================================================
# LOAD LOGISTIC REGRESSION ENSEMBLE
# ============================================================

@st.cache_resource
def load_ensemble_model():

    ensemble_path = os.path.join(
        MODEL_DIR,
        "ensemble_model.joblib"
    )

    if not os.path.exists(ensemble_path):

        raise FileNotFoundError(
            f"Ensemble model not found:\n{ensemble_path}"
        )

    return joblib.load(
        ensemble_path
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_uploaded_image(uploaded_file):

    img = Image.open(
        uploaded_file
    ).convert("RGB")

    img = img.resize(
        IMG_SIZE
    )

    img_array = np.array(
        img,
        dtype=np.float32
    )

    # IMPORTANT:
    # Your training generator used:
    #
    # rescale=1./255
    #
    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img, img_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(img_array):

    cnn_models = load_cnn_models()

    xgb_models = load_xgb_models()

    ensemble_model = load_ensemble_model()

    probabilities = []

    model_results = {}


    # --------------------------------------------------------
    # Each CNN + XGBoost
    # --------------------------------------------------------

    for model_name in [
        "resnet50",
        "densenet121",
        "vgg16",
        "inceptionv3"
    ]:

        cnn_model, preprocess_func = (
            cnn_models[model_name]
        )

        xgb_model = (
            xgb_models[model_name]
        )


        # CNN preprocessing
        processed_image = preprocess_func(
            img_array.copy()
        )


        # Feature extraction
        features = cnn_model.predict(
            processed_image,
            verbose=0
        )


        # XGBoost prediction
        dmatrix = xgb.DMatrix(
            features
        )

        probability = xgb_model.predict(
            dmatrix
        )[0]


        probabilities.append(
            probability
        )

        model_results[
            model_name
        ] = probability


    # --------------------------------------------------------
    # Stack four XGBoost probabilities
    # --------------------------------------------------------

    X_stack = np.array(
        probabilities
    ).reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # Logistic Regression ensemble
    # --------------------------------------------------------

    final_prediction = (
        ensemble_model.predict(
            X_stack
        )[0]
    )


    final_probabilities = (
        ensemble_model.predict_proba(
            X_stack
        )[0]
    )


    # Your folder structure produces:
    #
    # Glaucoma = 0
    # Normal = 1

    if final_prediction == 0:
        result = "Glaucoma"
    else:
        result = "Normal"

    confidence = np.max(
        final_probabilities
    )


    return (
        label,
        confidence,
        final_probabilities,
        model_results
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader("📤 Upload Fundus Image")

uploaded_file = st.file_uploader(
    "Choose a retinal fundus image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# DISPLAY IMAGE
# ============================================================

if uploaded_file is not None:

    image, img_array = (
        preprocess_uploaded_image(
            uploaded_file
        )
    )

    st.image(
        image,
        caption="Uploaded Fundus Image",
        use_container_width=True
    )


    st.markdown("---")


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    if st.button(
        "🔍 Detect Glaucoma",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Analyzing retinal image..."
            ):

                (
                    label,
                    confidence,
                    final_probabilities,
                    model_results
                ) = predict_image(
                    img_array
                )


            # =================================================
            # FINAL RESULT
            # =================================================

            st.markdown("---")

            st.subheader(
                "📊 Prediction Result"
            )


            if label == "Glaucoma":

                st.error(
                    f"⚠️ Prediction: {label}"
                )

            else:

                st.success(
                    f"✅ Prediction: {label}"
                )


            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )


            # =================================================
            # PROBABILITIES
            # =================================================

            st.subheader(
                "📈 Prediction Probabilities"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Glaucoma",
                    f"{final_probabilities[0]:.2%}"
                )


            with col2:

                st.metric(
                    "Normal",
                    f"{final_probabilities[1]:.2%}"
                )


            # =================================================
            # INDIVIDUAL MODEL RESULTS
            # =================================================

            st.subheader(
                "🤖 Individual XGBoost Predictions"
            )


            for model_name, probability in (
                model_results.items()
            ):

                st.write(
                    f"**{model_name.upper()}**"
                )

                st.progress(
                    float(probability)
                )

                st.write(
                    f"Glaucoma probability: "
                    f"{probability:.2%}"
                )


            # =================================================
            # DISCLAIMER
            # =================================================

            st.markdown("---")

            st.warning(
                """
                ⚠️ Medical Disclaimer

                This application is intended for
                educational and research purposes only.
                It is not a medical diagnostic tool.

                Please consult a qualified healthcare
                professional for actual medical diagnosis.
                """
            )


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)