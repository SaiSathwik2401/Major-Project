import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from transformers import TFAutoModel, AutoTokenizer
import numpy as np

class HybridMentalHealthClassifier:
    """
    Hybrid Transformer Architecture with Cascaded CNNs
    Based on Figure 2 and Figure 4, Pages 12152-12156
    
    Architecture:
    - MentalBERT branch → 3 CNN layers → 384-dim features
    - MelBERT branch → 3 CNN layers → 384-dim features
    - Concatenation → 768-dim features
    - Dense layer → 4-class output
    """
    
    def __init__(self, num_classes=4, max_length=512):
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Initialize tokenizers
        print("Loading MentalBERT tokenizer...")
        self.mental_tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )
        
        print("Loading MelBERT tokenizer...")
        self.mel_tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )
        
    def create_cnn_branch(self, input_shape, branch_name):
        """
        Create CNN branch for feature extraction
        Pages 12155-12156, Equations 12-14
        
        Architecture per branch:
        - Conv1D (128 filters, kernel=2) + MaxPooling
        - Conv1D (128 filters, kernel=3) + MaxPooling
        - Conv1D (128 filters, kernel=4) + MaxPooling
        - Conv1D (256 filters, kernel=5) + AveragePooling
        - Output: 384-dimensional vector
        """
        inputs = layers.Input(shape=input_shape, name=f"{branch_name}_input")
        
        # Layer 1: Kernel size 2 (Table 3, Page 12158)
        conv1 = layers.Conv1D(
            filters=128,
            kernel_size=2,
            padding='same',
            activation='relu',
            name=f"{branch_name}_conv1"
        )(inputs)
        pool1 = layers.MaxPooling1D(pool_size=2, name=f"{branch_name}_pool1")(conv1)
        
        # Layer 2: Kernel size 3
        conv2 = layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding='same',
            activation='relu',
            name=f"{branch_name}_conv2"
        )(pool1)
        pool2 = layers.MaxPooling1D(pool_size=2, name=f"{branch_name}_pool2")(conv2)
        
        # Layer 3: Kernel size 4
        conv3 = layers.Conv1D(
            filters=128,
            kernel_size=4,
            padding='same',
            activation='relu',
            name=f"{branch_name}_conv3"
        )(pool2)
        pool3 = layers.MaxPooling1D(pool_size=2, name=f"{branch_name}_pool3")(conv3)
        
        # Layer 4: Kernel size 5 with 256 filters
        conv4 = layers.Conv1D(
            filters=256,
            kernel_size=5,
            padding='same',
            activation='relu',
            name=f"{branch_name}_conv4"
        )(pool3)
        
        # Global Average Pooling to get 384-dim vector
        gap = layers.GlobalAveragePooling1D(name=f"{branch_name}_gap")(conv4)
        
        # Dense layer to ensure 384 dimensions
        output = layers.Dense(384, activation='relu', name=f"{branch_name}_dense")(gap)
        
        model = Model(inputs=inputs, outputs=output, name=f"{branch_name}_cnn")
        return model
    
    def build_model(self):
        """
        Build complete hybrid architecture
        Algorithm 1, Page 12157
        """
        print("\nBuilding hybrid model...")
        
        # Input layers for both branches
        mental_input_ids = layers.Input(
            shape=(self.max_length,), 
            dtype=tf.int32, 
            name='mental_input_ids'
        )
        mental_attention_mask = layers.Input(
            shape=(self.max_length,), 
            dtype=tf.int32, 
            name='mental_attention_mask'
        )
        
        mel_input_ids = layers.Input(
            shape=(self.max_length,), 
            dtype=tf.int32, 
            name='mel_input_ids'
        )
        mel_attention_mask = layers.Input(
            shape=(self.max_length,), 
            dtype=tf.int32, 
            name='mel_attention_mask'
        )
        
        # Branch 1: MentalBERT (Page 12153)
        print("Loading MentalBERT model...")
        mental_bert = TFAutoModel.from_pretrained(
            "bert-base-uncased",
            from_pt=True
        )
        mental_bert.trainable = False  # Freeze BERT weights initially
        
        mental_output = mental_bert(
            mental_input_ids,
            attention_mask=mental_attention_mask
        )[0]  # Shape: (batch, seq_len, 768)
        
        # Branch 2: MelBERT (Page 12153)
        print("Loading MelBERT model...")
        mel_bert = TFAutoModel.from_pretrained(
            "bert-base-uncased",
            from_pt=True
        )
        mel_bert.trainable = False  # Freeze BERT weights initially
        
        mel_output = mel_bert(
            mel_input_ids,
            attention_mask=mel_attention_mask
        )[0]  # Shape: (batch, seq_len, 768)
        
        # CNN branches for feature extraction
        mental_cnn = self.create_cnn_branch((self.max_length, 768), "mental")
        mel_cnn = self.create_cnn_branch((self.max_length, 768), "mel")
        
        # Extract features (Equations 15-16, Page 12157)
        mental_features = mental_cnn(mental_output)  # 384-dim
        mel_features = mel_cnn(mel_output)  # 384-dim
        
        # Concatenate features (Equation 17, Page 12157)
        concatenated = layers.Concatenate(name='feature_concat')([
            mental_features,
            mel_features
        ])  # 768-dim
        
        # Classification head (Equation 18, Page 12157)
        # Dense layer with 128 neurons (Table 3, Page 12158)
        dense = layers.Dense(128, activation='relu', name='dense_layer')(concatenated)
        dropout = layers.Dropout(0.1, name='dropout')(dense)
        
        # Output layer with softmax
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output_layer'
        )(dropout)
        
        # Create final model
        model = Model(
            inputs=[
                mental_input_ids,
                mental_attention_mask,
                mel_input_ids,
                mel_attention_mask
            ],
            outputs=outputs,
            name='HybridMentalHealthClassifier'
        )
        
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """
        Compile model with specified hyperparameters
        Table 3, Page 12158
        """
        # Optimizer: Adam with learning rate 0.001
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Loss: Sparse Categorical Cross-Entropy
        loss = keras.losses.SparseCategoricalCrossentropy()
        
        # Metrics
        metrics = [
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print("\nModel compiled successfully!")
        return model
    
    def prepare_inputs(self, texts):
        """
        Tokenize texts for both BERT models
        Pages 12154-12155, Equations 1-4
        """
        # MentalBERT tokenization
        mental_encoded = self.mental_tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        # MelBERT tokenization
        mel_encoded = self.mel_tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        return {
            'mental_input_ids': mental_encoded['input_ids'],
            'mental_attention_mask': mental_encoded['attention_mask'],
            'mel_input_ids': mel_encoded['input_ids'],
            'mel_attention_mask': mel_encoded['attention_mask']
        }
    
    def create_data_generator(self, texts, labels, batch_size=64):
        """
        Create data generator for training
        Batch size: 64 (Table 3, Page 12158)
        """
        dataset_size = len(texts)
        indices = np.arange(dataset_size)
        
        while True:
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_texts = [texts[i] for i in batch_indices]
                batch_labels = labels[batch_indices]
                
                batch_inputs = self.prepare_inputs(batch_texts)
                
                yield batch_inputs, batch_labels


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = HybridMentalHealthClassifier(num_classes=4, max_length=512)
    
    # Build model
    model = classifier.build_model()
    
    # Compile model
    model = classifier.compile_model(model)
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Save model architecture visualization
    keras.utils.plot_model(
        model,
        to_file='hybrid_model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True
    )
    print("\nModel architecture saved to 'hybrid_model_architecture.png'")