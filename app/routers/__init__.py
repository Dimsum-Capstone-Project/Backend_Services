from app.ml_utils.predict import PalmPrintRecognizer

recognizer = PalmPrintRecognizer(
    model_path="app/ml_utils/model/v2/palm_print_siamese_model.h5"
)
