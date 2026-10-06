import pandas as pd
import numpy as np

"""
Sample Mental Health Dataset
Based on Figure 1 (Page 12151) and Section III.A (Pages 12150-12152)

This creates a representative sample dataset similar to what's used in the paper.
Real datasets come from:
- Reddit r/depression, r/anxiety, r/BPD subreddits
- HuggingFace PTSD dataset
"""

# Sample texts for each mental illness category
# Based on actual linguistic patterns from the paper

depression_samples = [
    "I feel so empty inside, nothing brings me joy anymore. Every day is a struggle.",
    "Can't remember the last time I felt happy. Everything seems pointless and gray.",
    "I'm so tired all the time, but I can't sleep. Nothing matters anymore.",
    "Nobody would miss me if I was gone. I'm just a burden to everyone.",
    "I used to love painting but now I can't even pick up a brush. What's the point?",
    "The days just blend together. Wake up, feel numb, go back to sleep.",
    "I've been crying for hours and I don't even know why anymore.",
    "Food has no taste. Music has no meaning. Colors look dull. Is this living?",
    "My friends keep inviting me out but I just want to stay in bed forever.",
    "I look at old photos of myself smiling and wonder where that person went.",
    "Everything feels heavy. Even breathing feels like too much effort.",
    "I can't concentrate on anything. My mind is just foggy and dark all the time.",
    "People say it gets better but I've been waiting for years. Nothing changes.",
    "I feel like I'm watching my life from outside my body. Nothing feels real.",
    "The guilt is crushing me. I've let everyone down and I can't fix it.",
]

anxiety_samples = [
    "I can't stop worrying about everything. My heart races constantly and I can't breathe.",
    "What if something terrible happens? I keep checking the locks over and over.",
    "I feel like everyone is judging me. I can't go to social events anymore.",
    "My chest is tight and I'm sweating but the doctors say I'm fine. How can this be anxiety?",
    "I had three panic attacks today. I'm terrified of leaving the house.",
    "What if I fail the exam? What if I lose my job? What if, what if, what if?",
    "I've been awake for 48 hours because I can't turn off my racing thoughts.",
    "Every notification on my phone makes my heart jump. What if it's bad news?",
    "I rehearse conversations in my head 100 times before I make a simple phone call.",
    "The world feels like it's closing in on me. I need to escape but there's nowhere to go.",
    "I check my email obsessively because I'm terrified I missed something important.",
    "What if I said something wrong yesterday? Everyone probably hates me now.",
    "I can feel my heart pounding in my chest even when I'm just sitting still.",
    "I avoid driving because what if I cause an accident? The thought paralyzes me.",
    "I spent 2 hours planning a 10-minute grocery trip to minimize my anxiety.",
]

bpd_samples = [
    "I love my partner more than anything but right now I hate them so much I can't stand it.",
    "Everyone always leaves me. I must be fundamentally unlovable and broken.",
    "One minute I'm on top of the world, the next I want to disappear forever.",
    "I feel so empty, like there's a void inside me that nothing can fill.",
    "My friend didn't text me back fast enough so obviously they hate me now.",
    "I don't even know who I am anymore. My personality changes with whoever I'm around.",
    "I cut myself because the physical pain is better than the emotional agony.",
    "Why do I sabotage every good relationship? I push people away then beg them to stay.",
    "I feel like I'm too much for everyone. Too emotional, too needy, too intense.",
    "My mood swings are exhausting. I can't control how I feel from moment to moment.",
    "I idealize people then completely devalue them. Black and white, no in between.",
    "I'm terrified of abandonment but I also push everyone away. It makes no sense.",
    "Sometimes I feel like I'm multiple people trapped in one body.",
    "I react so intensely to everything. A small criticism feels like the end of the world.",
    "I make impulsive decisions when I'm upset and regret them immediately after.",
]

ptsd_samples = [
    "The flashbacks won't stop. I keep reliving that terrible day over and over.",
    "I can't go near that place anymore. Just thinking about it makes me panic.",
    "I wake up screaming from nightmares every single night. I'm exhausted.",
    "Loud noises make me jump and I'm instantly back in that moment again.",
    "I can't trust anyone anymore. The world feels dangerous and I'm always on edge.",
    "I avoid anything that reminds me of what happened but it's everywhere.",
    "I feel detached from everyone, like I'm behind a glass wall watching life.",
    "My hypervigilance is exhausting. I'm constantly scanning for threats.",
    "I blame myself every day for what happened even though it wasn't my fault.",
    "I can't talk about it. When I try, my throat closes up and I start shaking.",
    "Certain smells instantly transport me back and I lose all sense of present time.",
    "I've become numb. I can't feel happy or sad anymore, just empty and alert.",
    "I carry guilt like a weight on my chest. Why did I survive when others didn't?",
    "I avoid sleeping because the nightmares are too vivid and terrifying.",
    "I'm always irritable and angry now. It's like I'm a different person.",
]

# Create DataFrame
data = []

# Add all samples with labels
for text in depression_samples:
    data.append({'text': text, 'label': 'depression', 'label_encoded': 0})

for text in anxiety_samples:
    data.append({'text': text, 'label': 'anxiety', 'label_encoded': 1})

for text in bpd_samples:
    data.append({'text': text, 'label': 'bpd', 'label_encoded': 2})

for text in ptsd_samples:
    data.append({'text': text, 'label': 'ptsd', 'label_encoded': 3})

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display statistics
print("="*60)
print("SAMPLE MENTAL HEALTH DATASET")
print("="*60)
print(f"\nTotal Samples: {len(df)}")
print("\nClass Distribution:")
print(df['label'].value_counts())
print("\n" + "="*60)
print("\nFirst 5 Samples:")
print("="*60)
print(df.head())

# Save to CSV
df.to_csv('sample_mental_health_dataset.csv', index=False)
print("\n[SUCCESS] Dataset saved to 'sample_mental_health_dataset.csv'")

# Additional metadata about the dataset
metadata = {
    'source': 'Synthetic data based on research paper patterns',
    'paper': 'A Hybrid Transformer Architecture for Multiclass Mental Illness Prediction',
    'classes': ['Depression', 'Anxiety', 'BPD', 'PTSD'],
    'total_samples': len(df),
    'samples_per_class': len(df) // 4,
    'text_characteristics': {
        'depression': 'Feelings of emptiness, hopelessness, loss of interest',
        'anxiety': 'Excessive worry, fear, panic, hypervigilance',
        'bpd': 'Unstable relationships, emotional swings, identity issues',
        'ptsd': 'Flashbacks, nightmares, avoidance, hyperarousal'
    },
    'linguistic_features': {
        'depression': ['empty', 'pointless', 'numb', 'tired', 'burden'],
        'anxiety': ['worry', 'panic', 'what if', 'terrified', 'racing'],
        'bpd': ['love/hate', 'abandonment', 'empty', 'unstable', 'intense'],
        'ptsd': ['flashback', 'nightmare', 'reliving', 'triggered', 'avoid']
    }
}

print("\n" + "="*60)
print("DATASET METADATA")
print("="*60)
for key, value in metadata.items():
    if isinstance(value, dict):
        print(f"\n{key.upper()}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    else:
        print(f"\n{key}: {value}")

# Create expanded dataset (for actual training you'd need much more data)
print("\n" + "="*60)
print("CREATING EXPANDED DATASET FOR TRAINING")
print("="*60)

# In the paper, they use 10,000 samples per class (Table 2, Page 12153)
# Here we'll show how to structure it

expanded_data = []
target_samples_per_class = 100  # Reduced for demonstration

for label_name, label_code in [('depression', 0), ('anxiety', 1), ('bpd', 2), ('ptsd', 3)]:
    class_samples = df[df['label'] == label_name]['text'].tolist()
    
    # In real scenario, you would have 10,000 unique samples
    # Here we're just replicating for structure demonstration
    while len(expanded_data) < (label_code + 1) * target_samples_per_class:
        for text in class_samples:
            if len(expanded_data) < (label_code + 1) * target_samples_per_class:
                expanded_data.append({
                    'text': text,
                    'label': label_name,
                    'label_encoded': label_code
                })

expanded_df = pd.DataFrame(expanded_data)
expanded_df = expanded_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nExpanded Dataset Size: {len(expanded_df)}")
print("\nExpanded Class Distribution:")
print(expanded_df['label'].value_counts())

expanded_df.to_csv('expanded_mental_health_dataset.csv', index=False)
print("\n[SUCCESS] Expanded dataset saved to 'expanded_mental_health_dataset.csv'")

# Text statistics
print("\n" + "="*60)
print("TEXT STATISTICS")
print("="*60)

df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

print(f"\nAverage text length: {df['text_length'].mean():.2f} characters")
print(f"Average word count: {df['word_count'].mean():.2f} words")
print(f"\nText length by class:")
print(df.groupby('label')['text_length'].mean().round(2))
print(f"\nWord count by class:")
print(df.groupby('label')['word_count'].mean().round(2))

print("\n" + "="*60)
print("DATASET READY FOR PREPROCESSING!")
print("="*60)