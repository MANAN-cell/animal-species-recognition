# Animal Species Recognition Using Deep Learning

## Project Overview
Animal Species Recognition is a deep learning project that classifies animal images into five categories: Cat, Dog, Elephant, Horse, and Lion. The project uses a trained image classification model and provides a web interface for testing predictions.

## Supported Classes
- Cat
- Dog
- Elephant
- Horse
- Lion

## Dataset
The dataset contains approximately 13,496 training images and 1,500 validation images, with 300 validation images per class.

The dataset is organized into training and validation folders. The complete dataset is available separately due to its large size.

Download the complete dataset here:

[Download Animal Dataset (ZIP)](https://drive.google.com/file/d/1koV77n6m5LJbiRB0703kHrNu9ULsqT6h/view?usp=sharing)

- **Training images:** 13,496
- **Validation images:** 1,500
- **Classes:** Cat, Dog, Elephant, Horse, Lion
- **Image size:** 224 × 224 pixels

## How to Train the Model

### 1. Download the Dataset
Download the dataset ZIP file from the Google Drive link above and extract it.

### 2. Arrange the Dataset
Make sure the extracted folders follow this structure:

```text
Animal_Dataset/
└── Animal/
    ├── train/
    │   ├── cat/
    │   ├── dog/
    │   ├── ELEPHANT/
    │   ├── HORSE/
    │   └── LION/
    └── val/
        ├── cat/
        ├── dog/
        ├── ELEPHANT/
        ├── HORSE/
        └── LION/
```

### 3. Install Dependencies

```bash
pip install tensorflow
```

### 4. Run Training

```bash
python train_model.py
```

**Note:** The training script saves the trained model and class names inside the `models/` folder.

## Model Details
- Model: MobileNetV2-based image classifier
- Input image size: 224 × 224
- Number of classes: 5
- Validation accuracy: 96.80%
- Saved model formats: Keras (`.keras`) and TensorFlow Lite (`.tflite`)

## Project Files
- `train_model.py` — model training code
- `app.py` — application/API code
- `index.html` — web interface
- `requirements.txt` — Python dependencies
- `animal_species_model.tflite` — trained TensorFlow Lite model

## Live Demo
Website: https://animal-species-recognition-ten.vercel.app/

Prediction API: https://manan6726.pythonanywhere.com/predict

## Source Code
GitHub Repository: https://github.com/MANAN-cell/animal-species-recognition

## Project Report
The project report documents the problem statement, dataset, methodology, model architecture, evaluation results, deployment, limitations, and future improvements.

## How to Run Locally
1. Clone this repository.
2. Install the required dependencies listed in `requirements.txt`.
3. Run the application using the instructions supported by the application code.
4. Open the web interface and test an animal image.

## Evaluation
The model achieved 96.80% validation accuracy on the validation dataset. Actual performance may vary on unseen images.

## Future Improvements
- Add more animal categories.
- Improve performance on difficult or unfamiliar images.
- Expand model evaluation using an independent test dataset.
- Improve the web interface and documentation.

## Author
Abdul Manan  
BS Computer Science  
Superior Gold Campus, Lahore
