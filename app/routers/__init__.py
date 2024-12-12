from app.ml_utils.predict_v4 import PalmPrintRecognizerV3

recognizer = PalmPrintRecognizerV3(
    model_weights_path="app/ml_utils/model/v4/palm_print_siamese_model_v3.h5",
    model_json_path="app/ml_utils/model/v4/palm_print_siamese_model_v3.json"
)
