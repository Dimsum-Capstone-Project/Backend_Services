import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Lambda
import cv2
import os
import json
from typing import Dict, List, Tuple
import time
from app.storage_utils import upload_to_gcs, get_from_gcs
from dotenv import load_dotenv

load_dotenv(override=True)

def build_base_network(input_shape=(128, 128, 1)):
    """Build the base network for feature extraction"""
    inputs = Input(shape=input_shape)
    
    # First block
    x = Conv2D(64, (10, 10), activation='relu')(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (7, 7), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (4, 4), activation='relu')(x)
    
    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(4096, activation='sigmoid')(x)
    
    return Model(inputs, x)

class PalmPrintRecognizer:
    def __init__(self, model_path: str):
        # Build base network
        base_network = build_base_network()
        
        # Create Siamese network
        input_a = Input(shape=(128, 128, 1))
        input_b = Input(shape=(128, 128, 1))
        
        # Get embeddings
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        # Add Lambda layer to compute L1 distance
        L1_layer = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([processed_a, processed_b])
        
        # Add Dense layer with sigmoid activation
        prediction = Dense(1, activation='sigmoid')(L1_distance)
        
        # Create model
        self.model = Model(inputs=[input_a, input_b], outputs=prediction)
        
        # Load weights
        self.model.load_weights(model_path)
        
        # Create embedding model
        self.embedding_model = Model(inputs=input_a, outputs=processed_a)
        
        # Initialize embedding database
        self.embedding_db: Dict[str, List[np.ndarray]] = {}
        
    def reset_database(self):
        self.embedding_db = {}

    def preprocess_image(self, image):
        """
        Preprocess an image for the model. Can accept either a file path or a numpy array.
        """
        if isinstance(image, str):
            if os.getenv("USE_GCS", "false") == "true":
                content = get_from_gcs(image)
                img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                
            if img is None:
                raise ValueError(f"Could not load image from path: {image}")
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # img = image
            else:
                img = image
        else:
            raise ValueError("Image must be either a file path or numpy array")
            
        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=-1)

    def get_embedding(self, image):
        """
        Get embedding for an image. Can accept either a file path or preprocessed image array.
        """
        preprocessed_img = self.preprocess_image(image)
        preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
        # if isinstance(image, str):
        #     preprocessed_img = self.preprocess_image(image)
        #     preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
        # else:
            
        return self.embedding_model.predict(preprocessed_img)[0]  # Return flattened embedding
        

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
        
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute similarity between two embeddings using L1 distance
        """
        # Convert to numpy arrays if they aren't already
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Compute L1 distance
        l1_distance = np.sum(np.abs(emb1 - emb2))
        
        # Convert distance to similarity score (0 to 1)
        similarity = 1 / (1 + l1_distance)
        
        return similarity
    
    def compute_similarity2(self, embedding1, embedding2):
        """
        Compute similarity between two embeddings using Euclidean distance
        """
        # Convert to numpy arrays if they aren't already
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Compute Euclidean distance
        euclidean_distance = np.sqrt(np.sum((emb1 - emb2) ** 2))
        print(f"Euclidean distance: {euclidean_distance}")
        
        # Convert distance to similarity score (0 to 1) with steeper exponential decay
        similarity = np.exp(-2 * euclidean_distance)  # Increased decay factor
        
        return similarity
    
    def compute_distance(self, embedding1, embedding2):
        """
        Compute similarity between two embeddings using Euclidean distance
        """
        # Convert to numpy arrays if they aren't already
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Compute Euclidean distance
        euclidean_distance = np.sqrt(np.sum((emb1 - emb2) ** 2))
        print(f"Euclidean distance: {euclidean_distance}")
        
        return euclidean_distance
    
    def find_match2(self, image, threshold: float = 0.7) -> Tuple[str, float]:  
        """Find matching person in database with improved validation"""

        
        query_embedding = self.get_embedding(image)
        
        # Store all similarity scores
        similarities = []
        for person_id, embeddings_list in self.embedding_db.items():
            print(f"Person ID: {person_id}")
            for stored_embedding in embeddings_list:
                similarity = self.compute_similarity2(query_embedding, stored_embedding)
                similarities.append((person_id, similarity))
                print(f"Similarity: {similarity * 100:.2f}%")
        
        # # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        # get the smallest distance on the similarity dimension
        # similarities.sort(key=lambda x: x[1])

        # print(similarities)
        
        if not similarities:
            return None, 0.0
            
        best_match, best_similarity = similarities[0]

        print(f"Best match: {best_match} with similarity {best_similarity * 100:.2f}%")
        
        # Additional validation checks
        
        # 1. Check if similarity meets minimum threshold
        if best_similarity < threshold:
            print("Threshold not met")
            return None, best_similarity
            
        # 2. If we have more than one match, check confidence margin
        if len(similarities) > 1:
            second_best_similarity = similarities[1][1]
            margin = best_similarity - second_best_similarity

            print(f"Second best similarity: {second_best_similarity * 100:.2f}%")
            print(f"Margin: {margin:.5f}")

            
            # Require significant margin between best and second best match
            MIN_MARGIN = 0.15  # 15% minimum difference required
            if margin < MIN_MARGIN:
                print("Margin not met")
                return None, best_similarity
            
        # 3. Additional threshold check for very high confidence
        HIGH_CONFIDENCE_THRESHOLD = 0.85
        if best_similarity < HIGH_CONFIDENCE_THRESHOLD:
            # For matches below high confidence, require larger margin
            if len(similarities) > 1:
                margin = best_similarity - second_best_similarity
                if margin < MIN_MARGIN * 1.5:  # Require 50% more margin for lower confidence matches
                    print("High confidence threshold not met")
                    return None, best_similarity
        
        return best_match, best_similarity
    
    def find_match3(self, image, threshold: float = 290) -> Tuple[str, float]:  
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
            distances.append((person_id, distance_temp))
            print(f"distance_temp: {distance_temp}")
        
        # sort by distance


        distances.sort(key=lambda x: x[1])

        if not distances:
            return None, 0.0

        best_match, best_distance = distances[0]
        if best_distance > threshold:
            return None, best_distance
        
        return best_match, best_distance

    def find_match(self, image, threshold: float = 0.5) -> Tuple[str, float]:
        time_start = time.time()

        """Find matching person in database"""
        query_embedding = self.get_embedding(image)
        # print("find match")

        # print(query_embedding)
        
        best_match = None
        minimum_distance = float('inf')  # Initialize with infinity for Euclidean distance

        # print("------------------")

        # print(self.embedding_db)
        avg_distances = []
        for person_id, embeddings_list in self.embedding_db.items():
            # print("HERE 1")
            distances = []
            for stored_embedding in embeddings_list:
                # print("HERE 2")
                # Calculate Euclidean distance
                distance = np.linalg.norm(query_embedding - stored_embedding)
                # print(f"Distance: {distance} for {person_id}")
                distances.append(distance)
            avg_distance = np.mean(distances)
            avg_distances.append(avg_distance)
            # print(f"Average distance for {person_id}: {avg_distance}")

        index = np.argmin(avg_distances)
        min_val = avg_distances[index]
        best_match = list(self.embedding_db.keys())[index]

        print(f"min_val: {min_val}")


        if min_val > threshold:  # Adjust threshold logic for Euclidean distance
            return None, min_val
        
        time_end = time.time()
        print(f"Time taken: {time_end - time_start:.2f}s")
            
        return best_match, min_val

# Example usage
if __name__ == "__main__":
    # Initialize recognizer
    recognizer = PalmPrintRecognizer(
        model_path="models/palm_print_siamese_model.h5"
    )
    
    # Add some examples to database
    recognizer.add_to_database("person1", "data/person1_palm.jpg")
    recognizer.add_to_database("person2", "data/person2_palm.jpg")
    
    # Save database
    recognizer.save_database("palm_print_db.json")
    
    # Find match for new image
    person_id, similarity = recognizer.find_match("data/test_palm.jpg")
    if person_id:
        print(f"Match found: {person_id} with similarity {similarity * 100:.2f}%")
    else:
        print(f"No match found (similarity: {similarity * 100:.2f}%)")
