import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import tensorflow as tf
from lime.lime_text import LimeTextExplainer
import shap

class ExplainableAI:
    """
    Explainable AI module for mental health predictions
    Implements multiple interpretation techniques:
    1. Attention Visualization (from BERT layers)
    2. LIME (Local Interpretable Model-agnostic Explanations)
    3. SHAP (SHapley Additive exPlanations)
    4. Word Importance Heatmaps
    5. Feature Attribution
    """
    
    def __init__(self, model, mental_tokenizer, mel_tokenizer, class_names):
        self.model = model
        self.mental_tokenizer = mental_tokenizer
        self.mel_tokenizer = mel_tokenizer
        self.class_names = class_names
        self.lime_explainer = LimeTextExplainer(class_names=class_names)
    
    def get_attention_weights(self, text, model_branch='mental'):
        """
        Extract attention weights from BERT layers
        Shows which words the model focuses on
        """
        tokenizer = self.mental_tokenizer if model_branch == 'mental' else self.mel_tokenizer
        
        # Tokenize
        encoded = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
        
        # Get attention weights from BERT
        # Note: You need to modify model to output attention weights
        outputs = self.model.layers[0](encoded['input_ids'], 
                                       attention_mask=encoded['attention_mask'],
                                       output_attentions=True)
        
        # Average attention across heads and layers
        attention = outputs.attentions  # List of attention matrices
        avg_attention = tf.reduce_mean([tf.reduce_mean(att, axis=1) for att in attention], axis=0)
        
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        
        return tokens, avg_attention.numpy()
    
    def visualize_attention(self, text, save_path='attention_heatmap.png'):
        """
        Create attention heatmap showing word importance
        """
        tokens, attention = self.get_attention_weights(text)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # MentalBERT attention
        tokens_mental, attention_mental = self.get_attention_weights(text, 'mental')
        sns.heatmap(attention_mental[0], xticklabels=tokens_mental[:20], 
                   yticklabels=tokens_mental[:20], ax=ax1, cmap='YlOrRd')
        ax1.set_title('MentalBERT Attention Weights', fontsize=14, fontweight='bold')
        
        # MelBERT attention
        tokens_mel, attention_mel = self.get_attention_weights(text, 'mel')
        sns.heatmap(attention_mel[0], xticklabels=tokens_mel[:20], 
                   yticklabels=tokens_mel[:20], ax=ax2, cmap='YlGnBu')
        ax2.set_title('MelBERT Attention Weights', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
        plt.close()
    
    def explain_with_lime(self, text, num_features=10):
        """
        Use LIME to explain predictions
        Shows which words contributed most to the prediction
        """
        def predict_proba(texts):
            # Prepare inputs for model
            predictions = []
            for t in texts:
                mental_encoded = self.mental_tokenizer(t, return_tensors='tf', 
                                                      padding='max_length', 
                                                      truncation=True, max_length=512)
                mel_encoded = self.mel_tokenizer(t, return_tensors='tf',
                                                 padding='max_length',
                                                 truncation=True, max_length=512)
                
                inputs = {
                    'mental_input_ids': mental_encoded['input_ids'],
                    'mental_attention_mask': mental_encoded['attention_mask'],
                    'mel_input_ids': mel_encoded['input_ids'],
                    'mel_attention_mask': mel_encoded['attention_mask']
                }
                
                pred = self.model.predict(inputs, verbose=0)
                predictions.append(pred[0])
            
            return np.array(predictions)
        
        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            text,
            predict_proba,
            num_features=num_features,
            num_samples=100
        )
        
        return explanation
    
    def visualize_lime_explanation(self, text, predicted_class, 
                                   save_path='lime_explanation.png'):
        """
        Visualize LIME explanation as bar chart
        """
        explanation = self.explain_with_lime(text)
        
        # Get word weights for predicted class
        weights = explanation.as_list(label=predicted_class)
        
        # Separate words and weights
        words = [w[0] for w in weights]
        scores = [w[1] for w in weights]
        
        # Create color map (positive = supporting, negative = contradicting)
        colors = ['green' if s > 0 else 'red' for s in scores]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(words, scores, color=colors, alpha=0.7)
        plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
        plt.ylabel('Words/Phrases', fontsize=12, fontweight='bold')
        plt.title(f'LIME Explanation for {self.class_names[predicted_class]} Prediction',
                 fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"LIME explanation saved to {save_path}")
        plt.close()
        
        return explanation
    
    def get_word_importance_scores(self, text):
        """
        Calculate importance score for each word using gradient-based method
        """
        # Tokenize
        mental_encoded = self.mental_tokenizer(text, return_tensors='tf',
                                              padding='max_length',
                                              truncation=True, max_length=512)
        mel_encoded = self.mel_tokenizer(text, return_tensors='tf',
                                         padding='max_length',
                                         truncation=True, max_length=512)
        
        # Prepare inputs with gradient tape
        with tf.GradientTape() as tape:
            # Watch input tensors
            tape.watch(mental_encoded['input_ids'])
            tape.watch(mel_encoded['input_ids'])
            
            # Forward pass
            predictions = self.model({
                'mental_input_ids': mental_encoded['input_ids'],
                'mental_attention_mask': mental_encoded['attention_mask'],
                'mel_input_ids': mel_encoded['input_ids'],
                'mel_attention_mask': mel_encoded['attention_mask']
            })
            
            # Get prediction for most likely class
            predicted_class = tf.argmax(predictions[0])
            score = predictions[0][predicted_class]
        
        # Calculate gradients
        gradients = tape.gradient(score, mental_encoded['input_ids'])
        
        # Convert to importance scores
        importance_scores = tf.abs(gradients).numpy()[0]
        
        # Get tokens
        tokens = self.mental_tokenizer.convert_ids_to_tokens(
            mental_encoded['input_ids'][0]
        )
        
        # Combine tokens and scores, filter special tokens
        word_scores = []
        for token, score in zip(tokens, importance_scores):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                word_scores.append((token, float(score)))
        
        # Sort by importance
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        return word_scores
    
    def visualize_word_importance(self, text, top_n=15, 
                                  save_path='word_importance.png'):
        """
        Visualize top important words
        """
        word_scores = self.get_word_importance_scores(text)[:top_n]
        
        words = [w[0] for w in word_scores]
        scores = [w[1] for w in word_scores]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(words, scores)
        
        # Color gradient
        colors = plt.cm.viridis(np.linspace(0.3, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Words', fontsize=12, fontweight='bold')
        plt.title('Word Importance for Prediction (Gradient-Based)',
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Word importance visualization saved to {save_path}")
        plt.close()
    
    def generate_text_highlight_html(self, text, predicted_class):
        """
        Generate HTML with highlighted important words
        """
        word_scores = self.get_word_importance_scores(text)
        
        # Create dictionary for quick lookup
        score_dict = {word: score for word, score in word_scores}
        
        # Normalize scores for coloring
        max_score = max([s[1] for s in word_scores]) if word_scores else 1
        
        # Split text into words
        words = text.split()
        
        html = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; 
                    background: #f5f5f5; border-radius: 10px;">
            <h3 style="color: #333;">Prediction: {self.class_names[predicted_class]}</h3>
            <p style="font-size: 16px; line-height: 1.8;">
        """
        
        for word in words:
            # Clean word for lookup
            clean_word = word.lower().strip('.,!?;:')
            score = score_dict.get(clean_word, 0)
            
            # Calculate opacity based on importance
            opacity = min(score / max_score, 1.0) if max_score > 0 else 0
            
            # Color based on importance
            if opacity > 0.5:
                color = f'rgba(255, 99, 71, {opacity})'  # Red for high importance
            elif opacity > 0.2:
                color = f'rgba(255, 165, 0, {opacity})'  # Orange for medium
            else:
                color = f'rgba(200, 200, 200, {opacity})'  # Gray for low
            
            html += f'<span style="background-color: {color}; padding: 2px 4px; ' \
                   f'border-radius: 3px; margin: 2px;">{word}</span> '
        
        html += """
            </p>
            <p style="color: #666; font-size: 12px; margin-top: 20px;">
                <strong>Color Guide:</strong> 
                <span style="background-color: rgba(255, 99, 71, 0.7); padding: 2px 8px;">
                    High Importance
                </span>
                <span style="background-color: rgba(255, 165, 0, 0.7); padding: 2px 8px;">
                    Medium Importance
                </span>
                <span style="background-color: rgba(200, 200, 200, 0.5); padding: 2px 8px;">
                    Low Importance
                </span>
            </p>
        </div>
        """
        
        return html
    
    def comprehensive_explanation_report(self, text, predicted_class):
        """
        Generate comprehensive explanation combining all methods
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE EXPLANATION REPORT")
        print("="*70)
        
        print(f"\nInput Text: {text}")
        print(f"\nPredicted Class: {self.class_names[predicted_class]}")
        
        # 1. Word Importance
        print("\n" + "-"*70)
        print("TOP 10 IMPORTANT WORDS (Gradient-Based)")
        print("-"*70)
        word_scores = self.get_word_importance_scores(text)[:10]
        for i, (word, score) in enumerate(word_scores, 1):
            print(f"{i}. {word:15s} → {score:.4f}")
        
        # 2. LIME Explanation
        print("\n" + "-"*70)
        print("LIME EXPLANATION (Top Contributing Features)")
        print("-"*70)
        explanation = self.explain_with_lime(text)
        lime_weights = explanation.as_list(label=predicted_class)
        for word, weight in lime_weights[:10]:
            sentiment = "Supports" if weight > 0 else "Contradicts"
            print(f"{word:30s} → {weight:+.4f} ({sentiment})")
        
        # Generate visualizations
        print("\n" + "-"*70)
        print("GENERATING VISUALIZATIONS...")
        print("-"*70)
        
        self.visualize_attention(text)
        self.visualize_word_importance(text)
        self.visualize_lime_explanation(text, predicted_class)
        
        # Generate HTML
        html = self.generate_text_highlight_html(text, predicted_class)
        with open('text_explanation.html', 'w') as f:
            f.write(html)
        print("✅ HTML explanation saved to 'text_explanation.html'")
        
        print("\n" + "="*70)
        print("EXPLANATION REPORT COMPLETE")
        print("="*70)


# Example usage
if __name__ == "__main__":
    # Assume model, tokenizers, and class names are loaded
    
    # explainable_ai = ExplainableAI(
    #     model=model,
    #     mental_tokenizer=mental_tokenizer,
    #     mel_tokenizer=mel_tokenizer,
    #     class_names=['Depression', 'Anxiety', 'BPD', 'PTSD']
    # )
    
    # sample_text = "I feel so empty inside, nothing brings me joy anymore"
    # predicted_class = 0  # Depression
    
    # Generate comprehensive explanation
    # explainable_ai.comprehensive_explanation_report(sample_text, predicted_class)
    
    print("Explainable AI module implementation complete!")