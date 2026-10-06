import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

class ModelTrainer:
    """
    Training and evaluation pipeline
    Based on Section IV (Pages 12158-12159) and Section V (Pages 12159-12164)
    """
    
    def __init__(self, model, classifier, class_names):
        self.model = model
        self.classifier = classifier
        self.class_names = class_names
        self.history = None
    
    def create_callbacks(self, checkpoint_path='best_model.h5'):
        """
        Create training callbacks
        Page 12158 - 40 epochs with early stopping
        """
        callbacks = [
            # Save best model
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        return callbacks
    
    def train(self, train_texts, train_labels, val_texts, val_labels,
              epochs=40, batch_size=64):
        """
        Train the model
        Table 3, Page 12158:
        - Epochs: 40
        - Batch size: 64
        - Learning rate: 0.001
        """
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        
        # Prepare training data
        print("\nPreparing training data...")
        train_inputs = self.classifier.prepare_inputs(train_texts)
        train_labels = np.array(train_labels)
        
        # Prepare validation data
        print("Preparing validation data...")
        val_inputs = self.classifier.prepare_inputs(val_texts)
        val_labels = np.array(val_labels)
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        print(f"\nTraining for {epochs} epochs...")
        self.history = self.model.fit(
            train_inputs,
            train_labels,
            validation_data=(val_inputs, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)
        
        return self.history
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        Plot training and validation curves
        Figure 5, Page 12160
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved to {save_path}")
        plt.close()
    
    def evaluate(self, test_texts, test_labels):
        """
        Evaluate model on test set
        Pages 12158-12159, Equations 19-23
        """
        print("\n" + "="*50)
        print("Evaluating Model on Test Set")
        print("="*50)
        
        # Prepare test data
        test_inputs = self.classifier.prepare_inputs(test_texts)
        test_labels = np.array(test_labels)
        
        # Get predictions
        predictions = self.model.predict(test_inputs, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        print("\n" + "-"*50)
        print("Classification Report")
        print("-"*50)
        report = classification_report(
            test_labels,
            predicted_classes,
            target_names=self.class_names,
            digits=4
        )
        print(report)
        
        # Overall metrics
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            test_inputs,
            test_labels,
            verbose=0
        )
        
        # Calculate F1-score (Equation 22, Page 12159)
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        print("\n" + "-"*50)
        print("Overall Metrics")
        print("-"*50)
        print(f"Accuracy:  {test_acc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall:    {test_recall:.4f}")
        print(f"F1-Score:  {f1_score:.4f}")
        
        return {
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'test_labels': test_labels,
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': f1_score
        }
    
    def plot_confusion_matrix(self, test_labels, predicted_classes, 
                             save_path='confusion_matrix.png'):
        """
        Plot confusion matrix
        Figure 6a, Page 12160, Table 4
        """
        # Calculate confusion matrix
        cm = confusion_matrix(test_labels, predicted_classes)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {save_path}")
        plt.close()
        
        # Print per-class statistics
        print("\n" + "-"*50)
        print("Per-Class Statistics from Confusion Matrix")
        print("-"*50)
        
        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            accuracy = (tp + tn) / cm.sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n{class_name}:")
            print(f"  True Positives:  {tp}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
    
    def plot_roc_curve(self, test_labels, predictions, save_path='roc_curve.png'):
        """
        Plot ROC curves for multiclass classification
        Figure 6b, Page 12160
        """
        # Binarize labels for multiclass ROC
        test_labels_binarized = label_binarize(
            test_labels,
            classes=range(len(self.class_names))
        )
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(test_labels_binarized[:, i], predictions[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f'{class_name} (AUC = {roc_auc:.2f})'
            )
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - Multiclass Classification', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nROC curve saved to {save_path}")
        plt.close()
    
    def save_results_to_csv(self, results, save_path='evaluation_results.csv'):
        """
        Save evaluation results to CSV
        """
        results_df = pd.DataFrame({
            'True_Label': results['test_labels'],
            'Predicted_Label': results['predicted_classes'],
            'Confidence': np.max(results['predictions'], axis=1)
        })
        
        results_df.to_csv(save_path, index=False)
        print(f"\nResults saved to {save_path}")


# Example usage
if __name__ == "__main__":
    # Assume model and classifier are already created
    # from previous steps
    
    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv('train_data.csv')
    val_df = pd.read_csv('val_data.csv')
    test_df = pd.read_csv('test_data.csv')
    
    train_texts = train_df['text_clean'].tolist()
    train_labels = train_df['label_encoded'].tolist()
    
    val_texts = val_df['text_clean'].tolist()
    val_labels = val_df['label_encoded'].tolist()
    
    test_texts = test_df['text_clean'].tolist()
    test_labels = test_df['label_encoded'].tolist()

    class_names = ['Depression', 'Anxiety', 'BPD', 'PTSD']
    
    # Initialize classifier
    from hybrid_classifier import HybridMentalHealthClassifier
    classifier = HybridMentalHealthClassifier(num_classes=4, max_length=128) # Reduced length for faster demo
    
    # Build model
    model = classifier.build_model()
    model = classifier.compile_model(model)
    
    # Initialize trainer
    trainer = ModelTrainer(model, classifier, class_names)
    
    # Train model
    history = trainer.train(
        train_texts, train_labels,
        val_texts, val_labels,
        epochs=3, # Reduced epochs for demo
        batch_size=8 # Reduced batch size for small data
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate on test set
    results = trainer.evaluate(test_texts, test_labels)
    
    # Save model
    model.save('trained_hybrid_model.h5')
    print("\nModel saved to 'trained_hybrid_model.h5'")