# Speech Emotion Recognition

## Description

This project performs **speech emotion recognition** using the **Wav2Vec2 transformer model** from **Hugging Face**, trained and tested in **Google Colab**.  
The model classifies speech audio clips into one of seven emotional categories: *angry, disgust, fear, happy, neutral, pleasant surprise, sad*.

**❗❗ To view the full project, including code, metrics, and visualizations, download the following files and open them locally in any web browser: ❗❗**

- `Model_Training.html` – complete model training notebook  
- `Run_Model.html` – model testing and evaluation notebook

## How the Model Works

1. **Dataset Preparation**: The [Toronto emotional speech set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) was extracted, organized, and split into 80% training and 20% testing sets.  
2. **Feature Extraction**: Raw audio signals were processed with the `Wav2Vec2Processor` to obtain normalized input features.  
3. **Model Training**: A `Wav2Vec2ForSequenceClassification` model was fine-tuned for emotion classification using PyTorch.  
4. **Evaluation**: Performance metrics such as accuracy, precision, recall, and F1-score were computed on the test set.  
5. **Visualization**: Confusion matrix and probability plots were generated to analyze predictions.

## Technologies Used

- Python (**Google Colab**)  
- Hugging Face Transformers  
- PyTorch  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- scikit-learn  

## Model Results

| Epoch | Validation Loss | Accuracy | Precision | Recall | F1 |
|-------|-----------------|-----------|-----------|--------|------|
| 1 | 0.783 | 0.873 | 0.899 | 0.873 | 0.858 |
| 2 | 0.357 | 0.982 | 0.984 | 0.982 | 0.982 |
| 3 | 0.263 | 0.996 | 0.997 | 0.996 | 0.996 |

**Confusion Matrix and Prediction Visualization:**  
Displayed in `Run_Model.html`, showing model confidence and class distribution across test samples.

## Project Files

- `Model_Training.ipynb` – training and saving the model  
- `Run_Model.ipynb` – model loading, testing, and result visualization  
- `MelinteCosmin_SpeechEmotionRecognition.pptx` – project presentation  

## Disclaimer

This project was developed for academic purposes. You may use or adapt it for learning or research, but please avoid submitting it as your own work in coursework or formal assessments.
