import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Union
from app.ml_utils.preprocessing.palm_processor_enhanced import PalmPreprocessor
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Lambda
import json
from google.cloud import storage
from app.storage_utils import upload_to_gcs, get_from_gcs
from dotenv import load_dotenv
from typing import Dict, List

load_dotenv(override=True)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Contrastive loss function.
    
    Args:
        y_true: Label (1 for same pairs, 0 for different pairs)
        y_pred: Predicted distance
        margin: Margin for negative pairs
    """
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def build_feature_extractor(input_shape=(128, 128, 1)):
    inputs = Input(shape=input_shape)
    
    # Layer 1: Konvolusi 64 filter 10x10
    x = Conv2D(64, (10, 10), activation='relu', name='conv2d')(inputs)
    x = MaxPooling2D()(x)
    
    # Layer 2: Konvolusi 128 filter 7x7
    x = Conv2D(128, (7, 7), activation='relu', name='conv2d_1')(x)
    x = MaxPooling2D()(x)
    
    # Layer 3: Konvolusi 128 filter 4x4
    x = Conv2D(128, (4, 4), activation='relu', name='conv2d_2')(x)
    x = MaxPooling2D()(x)
    
    # Layer 4: Konvolusi 256 filter 4x4
    x = Conv2D(256, (4, 4), activation='relu', name='conv2d_3')(x)
    x = MaxPooling2D()(x)
    
    # Flatten dan Dense layer
    x = Flatten()(x)
    x = Dense(4096, activation='sigmoid', name='dense1')(x)
    
    # Normalisasi L2
    x = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1))(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model

def build_siamese_network():
    # Build feature extractor
    feature_extractor = build_feature_extractor()
    
    # Create siamese network
    input_a = Input(shape=(128, 128, 1), name='input_layer_1')
    input_b = Input(shape=(128, 128, 1), name='input_layer_2')
    
    # Share weights between twin networks
    encoded_a = feature_extractor(input_a)
    encoded_b = feature_extractor(input_b)
    
    # Add L2 distance between the embeddings
    l2_distance = Lambda(euclidean_distance, name='lambda')([encoded_a, encoded_b])
    
    # Create the siamese network
    model = Model(inputs=[input_a, input_b], outputs=l2_distance, name='functional')
    
    # Compile model with contrastive loss
    model.compile(loss=contrastive_loss, optimizer='adam', metrics=['accuracy'])
    
    return model, feature_extractor

class PalmPrintRecognizerV3:
    def __init__(self, model_json_path: str, model_weights_path: str):
        self.embedding_db: Dict[str, List[np.ndarray]] = {}
        try:
            print(f"\nLoading model from:\nJSON: {model_json_path}\nWeights: {model_weights_path}\n")
            
            # Initialize embedding database
            # Build model and feature extractor
            self.model, self.feature_extractor = build_siamese_network()
            
            # Load weights if file exists
            if os.path.exists(model_weights_path):
                try:
                    self.model.load_weights(model_weights_path)
                    logger.info("Model weights loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading model weights: {e}")
                    raise
            else:
                logger.warning(f"Weights file not found: {model_weights_path}")
            
            # Initialize preprocessor
            self.preprocessor = PalmPreprocessor()
            logger.info("Palm preprocessor initialized")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def add_to_database(self, person_id: str, image):
        """Add a new palm print to a person's database entry"""
        embedding = self.get_embedding(image)
        # print("add to database")
        # print(embedding)
        if person_id in self.embedding_db:
            self.embedding_db[person_id].append(embedding)
        else:
            self.embedding_db[person_id] = [embedding]

    def save_database_to_gcs(self, bucket_name: str, file_name: str) -> None:
        """Save embedding database to GCS"""
        db_serializable = {
            person_id: [embedding.tolist() for embedding in embeddings]
            for person_id, embeddings in self.embedding_db.items()
        }
        content = json.dumps(db_serializable)
        upload_to_gcs(file_name, content)
        
    def save_database(self, db_path: str) -> None:
        """Save embedding database to file"""
        # Convert numpy arrays to lists for JSON serialization
        db_serializable = {
            person_id: [embedding.tolist() for embedding in embeddings]
            for person_id, embeddings in self.embedding_db.items()
        }
        with open(db_path, 'w') as f:
            json.dump(db_serializable, f)

    def append_database(self, db_path: str) -> None:
        """Append embedding database to file"""
        # Load existing database if it exists
        if os.getenv("USE_GCS", "false") == "true":
            content = get_from_gcs(db_path)

            if content is None:
                db_serializable = {}
            else:
                db_serializable = json.loads(content)
        else:
            if os.path.exists(db_path):
                with open(db_path, 'r') as f:
                    db_serializable = json.load(f)
            else:
                db_serializable = {}

        # Update the database with new embeddings
        for person_id, embeddings in self.embedding_db.items():
            embeddings_serializable = [embedding.tolist() for embedding in embeddings]
            if person_id in db_serializable:
                db_serializable[person_id].extend(embeddings_serializable)
            else:
                db_serializable[person_id] = embeddings_serializable

        # Save the updated database
        if os.getenv("USE_GCS", "false") == "true":
            content = json.dumps(db_serializable)
            upload_to_gcs(db_path, content)
        else:
            with open(db_path, 'w') as f:
                json.dump(db_serializable, f)

            
    def load_database(self, db_path: str) -> None:
        """Load embedding database from file"""

        if os.getenv("USE_GCS", "false") == "true":
            content = get_from_gcs(db_path)

            # cek if content is json or not
            if content is None:
                db_dict = {}
            else:
                db_dict = json.loads(content)
        else:
            with open(db_path, 'r') as f:
                db_dict = json.load(f)

        # print(db_dict)
        # Convert lists back to numpy arrays
        self.embedding_db = {
            person_id: [np.array(embedding) for embedding in embeddings]
            for person_id, embeddings in db_dict.items()
        }
            
    def _preprocess_image(self, image):
        """Preprocess a palm print image."""
        try:
            return self.preprocessor.preprocess_image(image)
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
            
    def get_embedding(self, image):
        """Get embedding vector for a palm print image."""
        if isinstance(image, str):
            image = self._preprocess_image(image)
            
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        return self.feature_extractor.predict(image)[0]
        
    def compare_images(self, image1, image2):
        """Compare two palm print images and return similarity score and distance."""
        # Get embeddings
        embedding1 = self.get_embedding(image1)
        embedding2 = self.get_embedding(image2)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(embedding1 - embedding2)
        
        # Convert distance to similarity score
        similarity = np.exp(-1.5 * distance)
            
        return float(similarity), float(distance)
    
    def compute_distance(self, embedding1, embedding2):
        """Compute Euclidean distance between two embeddings."""

        return np.linalg.norm(embedding1 - embedding2)

    def compute_similarity(self, distance: float) -> float:
        """Compute similarity score from distance."""
        return np.exp(-1.5 * distance)
    
    
    def reset_database(self):
        self.embedding_db = {}
    
    def find_match3(self, image, threshold: float = 290, use_threshold = True) -> Tuple[str, float]:  
        """Find matching person in database with improved validation"""

        
        query_embedding = self.get_embedding(image)
        print("query_embedding")
        
        # Store all distances scores
        distances = []
        print("distances")
        for person_id, embeddings_list in self.embedding_db.items():
            print(f"Person ID: {person_id}")
            distance_temp = 0
            for stored_embedding in embeddings_list:
                distance = self.compute_distance(query_embedding, stored_embedding)
                distance_temp += distance
                print(f"distance: {distance}")
                break
            distances.append((person_id, distance_temp))
            print(f"distance_temp: {distance_temp}")
        
        # sort by distance


        distances.sort(key=lambda x: x[1])

        if not distances:
            return None, 0.0

        best_match, best_distance = distances[0]

        if use_threshold:
            if best_distance > threshold:
                return None, best_distance
        
        return best_match, best_distance
        
    def verify_palm(self, image1, image2, base_threshold=0.65):
        """Verify if two palm print images belong to the same person."""
        try:
            # Preprocess both images
            processed_img1 = self._preprocess_image(image1)
            processed_img2 = self._preprocess_image(image2)
            
            # Compare images
            similarity, distance = self.compare_images(processed_img1, processed_img2)
            
            # Adaptive threshold with finer granularity and balanced thresholds
            if distance < 0.14:
                threshold = base_threshold * 0.9    # 0.585 - Perfect match
            elif 0.14 <= distance < 0.145:
                threshold = base_threshold * 1.15   # 0.7475 - Very similar, likely same person
            elif 0.145 <= distance < 0.15:
                threshold = base_threshold * 1.231   # 0.8001 - Moderate for potential false negatives
            elif 0.15 <= distance < 0.155:
                threshold = base_threshold * 1.25   # 0.8125 - Strict for potential false positives
            elif 0.155 <= distance < 0.16:
                threshold = base_threshold * 1.3    # 0.845 - Very strict for high distance
            elif 0.16 <= distance < 0.19:
                threshold = base_threshold * 1.16   # 0.78 - Standard different person range
            else:
                threshold = base_threshold * 1.16   # 0.75 - Clearly different
                
            # Determine match based on threshold and apply distance penalty
            # Add small penalty for larger distances to help differentiate edge cases
            distance_penalty = max(0, (distance - 0.14) * 0.1)
            adjusted_similarity = similarity - distance_penalty
            
            is_match = adjusted_similarity >= threshold
            # print(f"adjusted_similarity: {adjusted_similarity}, threshold: {threshold}")
            
            logger.info(f"Verification Test Result: Match={is_match}, Similarity={similarity:.4f}, Adjusted={adjusted_similarity:.4f}, Distance={distance:.4f}, Threshold={threshold:.4f}")
            return is_match, similarity, distance
            
        except Exception as e:
            logger.error(f"Failed to preprocess one or both images")
            return False, 0.0, float('inf')


# Example usage
if __name__ == "__main__":
    # Initialize recognizer
    MODEL_JSON_PATH = "palm_print_siamese_model_v3.json"
    MODEL_WEIGHTS_PATH = "palm_print_siamese_model_v3.h5"
    
    recognizer = PalmPrintRecognizerV3(MODEL_JSON_PATH, MODEL_WEIGHTS_PATH)
    
    # Test verification
    test_image1 = "data_test/Image Registered.jpg"
    test_image2 = "data_test/Login Attempt.jpg"
    test_image3 = "data_test/diff1.jpg"
    test_image4 = "data_test/Flash Light Miring.jpg"
    test_image5 = "data_test/Flash Light Normal.jpg"
    
    
    
    if os.path.exists(test_image1) and os.path.exists(test_image2):
        is_match, similarity, distance = recognizer.verify_palm(test_image4, test_image5, base_threshold=0.55)
        print(f"Verification result: {'Match' if is_match else 'No match'}")
        print(f"Similarity score: {similarity:.4f}")
        print(f"Distance: {distance:.4f}")
    
    # # Test identification
    # test_db_path = "path/to/test/database.json"
    # if os.path.exists(test_db_path):
    #     recognizer.load_database(test_db_path)
    #     best_match, similarity = recognizer.identify_palm(test_image1)
    #     print(f"Best match: {best_match}")
    #     print(f"Similarity score: {similarity:.4f}")