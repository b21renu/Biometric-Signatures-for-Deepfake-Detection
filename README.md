# Biometric Signatures for Deepfake Detection

This project explores **biometric-based deepfake detection** by combining **audio and video analysis** to identify synthetic content. The goal was to develop a robust system that captures both **voice-based traits** and **facial behavioral cues** — subtle patterns that are hard for deepfake models to replicate.

We built:
- An **audio deepfake detector** using **BiLSTM** trained on MFCC, Chroma, and Tonnetz features.
- A **video deepfake detector** using **CNN-LSTM** to analyze spatial (frame-level) and temporal (sequence-level) inconsistencies like blinking, expression shifts, or erratic motion.

To enhance model generalization, we used **GANs** to augment synthetic data across both modalities.

---

## Results

| Model        | Modality | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|--------------|----------|----------|-----------|--------|----------|---------|
| KNN (Baseline) | Audio    | 78%      | 82%       | 70%    | 75%      | –       |
| BiLSTM       | Audio    | **92%**  | **93%**   | **91%**| **92%**  | –       |
| CNN-LSTM     | Video    | **92.5%**| **94.3%** | **91.7%**| **93.0%**| **0.965** |

---

## Tech Stack
- **Languages/Frameworks:** Python, PyTorch, TensorFlow
- **Libraries:** Librosa, OpenCV, Scikit-learn, Seaborn, Matplotlib
- **Data Sources:** VoxCeleb, LibriSpeech, FaceForensics++
