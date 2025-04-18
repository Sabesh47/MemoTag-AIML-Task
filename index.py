# ====== DURATION-ROBUST COGNITIVE DECLINE DETECTION ======
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ====== AUDIO PROCESSING ======
AUDIO_FOLDER = r'C:\Users\sabes\OneDrive\Desktop\abc\audio_samples'
all_features = []

for filename in os.listdir(AUDIO_FOLDER):
    if filename.endswith('.wav'):
        filepath = os.path.join(AUDIO_FOLDER, filename)
        print(f"Analyzing: {filename}...")
        
        try:
            y, sr = librosa.load(filepath, sr=16000)  # Standardize sample rate
            
            # ====== DURATION-INDEPENDENT FEATURES ======
            features = {'filename': filename}
            
            # 1. Pause characteristics (normalized per second)
            intervals = librosa.effects.split(y, top_db=25, frame_length=2048)
            pauses = [librosa.samples_to_time(i[1]-i[0], sr=sr) 
                     for i in intervals if (i[1]-i[0])/sr > 0.2]  # >200ms pauses
            
            features.update({
                'pauses_per_sec': len(pauses) / librosa.get_duration(y=y, sr=sr),
                'avg_pause_duration': np.mean(pauses) if pauses else 0,
                'pause_variability': np.std(pauses) if len(pauses) > 1 else 0
            })
            
            # 2. Pitch analysis (duration-independent)
            pitches = librosa.yin(y, fmin=80, fmax=400)  # Human vocal range
            pitches = pitches[(pitches > 0) & ~np.isnan(pitches)]
            features.update({
                'pitch_std': np.std(pitches) if len(pitches) > 0 else 0,
                'pitch_range': np.ptp(pitches) if len(pitches) > 0 else 0
            })
            
            # 3. Speech rhythm (normalized)
            rms = librosa.feature.rms(y=y, frame_length=2048)[0]
            voiced_frames = np.sum(rms > 0.02 * np.max(rms))
            features['speech_density'] = voiced_frames / len(rms)  # 0-1 scale
            
            all_features.append(features)
            
        except Exception as e:
            print(f"Skipped {filename}: {str(e)}")
            continue

# Create DataFrame
df = pd.DataFrame(all_features).set_index('filename')

# ====== ANOMALY DETECTION ======
# Select duration-independent features
X = df[['pauses_per_sec', 'avg_pause_duration', 'pitch_std', 'speech_density']]
X_scaled = StandardScaler().fit_transform(X)

# Optimized Isolation Forest
iso = IsolationForest(
    contamination='auto',  # Let algorithm determine threshold
    max_samples=0.8,       # Better for small datasets
    n_estimators=300,      # More trees for stability
    random_state=42
)
iso.fit(X_scaled)

# Generate results
df['anomaly_score'] = iso.decision_function(X_scaled)
df['is_anomaly'] = df['anomaly_score'] < np.percentile(df['anomaly_score'], 20)  # Flag bottom 20%

# Convert 'is_anomaly' to string for seaborn compatibility
df['is_anomaly'] = df['is_anomaly'].astype(str)

# ====== VISUALIZATION ======
plt.figure(figsize=(18, 6))
plt.suptitle("Cognitive Decline Analysis (Duration-Normalized)", y=1.05)

# Plot each feature
features_to_plot = ['pauses_per_sec', 'avg_pause_duration', 'pitch_std', 'speech_density']
for i, col in enumerate(features_to_plot, 1):
    plt.subplot(1, 4, i)
    
    sns.boxplot(
        data=df, 
        x='is_anomaly', 
        y=col,
        palette={'True': '#ff6b6b', 'False': '#74b9ff'},
        width=0.5,
        showfliers=False
    )
    sns.swarmplot(data=df, x='is_anomaly', y=col, color='.3', size=5)
    
    plt.xticks([0, 1], ['Normal', 'Potential Risk'])
    plt.title(col.replace('_', ' ').title())
    plt.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.show()

# ====== RESULTS INTERPRETATION ======
print("\n=== DURATION-INDEPENDENT RESULTS ===")
print(f"Processed {len(df)} files | {df['is_anomaly'].astype(str).value_counts().get('True', 0)} potential risk cases")

print("\nTop risk files:")
print(df[df['is_anomaly'] == 'True'].sort_values('anomaly_score')[['anomaly_score']])

print("\nKey feature differences:")
risk_stats = df.groupby('is_anomaly').mean()
print(risk_stats[features_to_plot])

# Save results
df.to_csv(os.path.join(AUDIO_FOLDER, 'cognitive_analysis_results.csv'))
print("\nResults saved to cognitive_analysis_results.csv")
