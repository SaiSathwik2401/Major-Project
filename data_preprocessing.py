import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class MentalHealthDataPreprocessor:
    """
    Data preprocessing pipeline for mental health text classification
    Based on methodology from pages 12153-12154
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.class_mapping = {
            'depression': 0,
            'anxiety': 1,
            'bpd': 2,
            'ptsd': 3
        }
    
    def clean_text(self, text):
        """
        Comprehensive text cleaning as per Figure 3, Page 12153
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters (keeping basic punctuation)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def handle_missing_values(self, df):
        """
        Handle missing values in dataset
        Page 12153 - Remove rows with missing text
        """
        print(f"Original dataset size: {len(df)}")
        
        # Remove rows with missing text
        df = df.dropna(subset=['text'])
        
        # Remove rows with missing labels
        df = df.dropna(subset=['label'])
        
        print(f"After removing missing values: {len(df)}")
        return df
    
    def remove_duplicates(self, df):
        """
        Remove duplicate entries to ensure data quality
        Page 12153
        """
        print(f"Before deduplication: {len(df)}")
        df = df.drop_duplicates(subset=['text'], keep='first')
        print(f"After deduplication: {len(df)}")
        return df
    
    def balance_dataset(self, df, samples_per_class=10000):
        """
        Random downsampling for balanced dataset
        Table 2, Page 12153 - 10,000 samples per class
        """
        balanced_dfs = []
        
        for label in df['label'].unique():
            class_df = df[df['label'] == label]
            
            if len(class_df) > samples_per_class:
                class_df = class_df.sample(n=samples_per_class, random_state=42)
            else:
                print(f"Warning: Class {label} has only {len(class_df)} samples")
            
            balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nBalanced dataset distribution:")
        print(balanced_df['label'].value_counts())
        
        return balanced_df
    
    def encode_labels(self, df):
        """
        Convert categorical labels to numerical
        Page 12153 - Depression=0, Anxiety=1, BPD=2, PTSD=3
        """
        # Map string labels to integers
        df['label_encoded'] = df['label'].map(self.class_mapping)
        
        # Verify all labels were mapped
        if df['label_encoded'].isna().any():
            print("Warning: Some labels couldn't be mapped!")
            print(df[df['label_encoded'].isna()]['label'].unique())
        
        return df
    
    def split_dataset(self, df, train_ratio=0.845, val_ratio=0.08, test_ratio=0.075):
        """
        Split dataset into train/val/test sets
        Table 2, Page 12153:
        - Training: 8,450 per class (84.5%)
        - Validation: 800 per class (8%)
        - Testing: 750 per class (7.5%)
        """
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1 - train_ratio),
            stratify=df['label_encoded'],
            random_state=42
        )
        
        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_df['label_encoded'],
            random_state=42
        )
        
        print(f"\nDataset split:")
        print(f"Training: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Testing: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def preprocess_pipeline(self, df):
        """
        Complete preprocessing pipeline
        Figure 3, Page 12153
        """
        print("Starting preprocessing pipeline...\n")
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Clean text
        print("\nCleaning text...")
        df['text_clean'] = df['text'].apply(self.clean_text)
        
        # Step 3: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 4: Remove empty texts after cleaning
        df = df[df['text_clean'].str.len() > 0]
        
        # Step 5: Balance dataset
        df = self.balance_dataset(df)
        
        # Step 6: Encode labels
        df = self.encode_labels(df)
        
        # Step 7: Split dataset
        train_df, val_df, test_df = self.split_dataset(df)
        
        print("\nPreprocessing complete!")
        return train_df, val_df, test_df


# Example usage
if __name__ == "__main__":
    # Sample data structure
    # Load the actual expanded dataset
    try:
        df = pd.read_csv('expanded_mental_health_dataset.csv')
    except FileNotFoundError:
        print("Dataset not found! Creating sample df...")
        sample_data = {
            'text': [
                "I feel so empty inside", "Can't stop worrying", 
                "Everyone leaves me", "The flashbacks won't stop"
            ] * 5,  # Multiply to have enough for split
            'label': ['depression', 'anxiety', 'bpd', 'ptsd'] * 5
        }
        df = pd.DataFrame(sample_data)
    
    preprocessor = MentalHealthDataPreprocessor()
    train_df, val_df, test_df = preprocessor.preprocess_pipeline(df)
    
    # Save processed data
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)