# MCOC Arena Bot - Action Predictor

An AI-powered system designed to predict actions in the game *Marvel Contest of Champions (MCOC)* using a hybrid Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) architecture.

## 🧠 Model Architecture

The system leverages a **CNN + RNN** architecture to classify real-time gameplay into one of five distinct actions:

- `light_attack` (a)
- `special` (s)
- `block` (d)
- `evade` (w)
- `medium_attack` (space)

### Model Structure

The model processes sequences of game frames to understand both the spatial content of each frame and the temporal relationship between them.

```
[Frame_1] [Frame_2] ... [Frame_10]
    ↓         ↓              ↓
[ CNN Backbone (MobileNetV2) ]
    ↓         ↓              ↓
[Feature_1] [Feature_2] ... [Feature_10]
                  ↓
            [2-Layer GRU]
                  ↓
          [Dense + Softmax]
                  ↓
           [Predicted Action]
```

**Key Features:**
-   **Input**: Sequences of 10 frames, representing 2.5 seconds of gameplay captured at 4 FPS.
-   **CNN Backbone**: A pre-trained MobileNetV2 is utilized for efficient feature extraction from each frame.
-   **RNN Core**: A 2-layer Gated Recurrent Unit (GRU) network processes the sequence of features to model temporal dependencies.
-   **Output**: The model outputs a probability distribution over the 5 action classes every 200-250ms.

## 📁 Project Structure

The repository is organized to separate data, models, source code, and results for clarity and maintainability.

```
mcoc-arena-bot/
├── assets/
│   ├── config/           # Game-related configurations
│   └── data/
│       └── raw/          # Labeled and numbered screenshots
├── models/               # Trained model checkpoints
├── results/              # Training plots and performance metrics
├── scripts/
│   ├── train_model.py    # Model training script
│   ├── inference.py      # Real-time inference script
│   └── test_model.py     # Model testing and analysis script
└── requirements.txt      # Python dependencies
```

## 🚀 Getting Started

Follow these steps to set up the project and begin training your own models.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd mcoc-arena-bot
```

### 2. Install Dependencies

It is recommended to create a virtual environment to manage dependencies.

```bash
pip install -r requirements.txt
```

### 3. Prepare the Data

-   Place your collected gameplay screenshots in the `assets/data/raw/` directory.
-   Files must be named sequentially with the action label appended after an underscore (e.g., `003631_a.png`).

## 🎯 Usage

### 1. Model Training

Navigate to the `scripts` directory and run the training script.

```bash
cd scripts
python train_model.py
```

**Configurable Parameters in `train_model.py`:**
-   `SEQUENCE_LENGTH`: The number of frames in each input sequence (default: 10).
-   `BATCH_SIZE`: The number of sequences per training batch (default: 16).
-   `LEARNING_RATE`: The learning rate for the optimizer (default: 1e-4).
-   `NUM_EPOCHS`: The total number of training epochs (default: 30).

**Outputs:**
-   The best performing model is saved to `models/best_model.pth`.
-   Training and validation performance graphs are saved in `results/`.
-   A comprehensive performance report is generated.

### 2. Model Evaluation

Execute the test script to analyze the trained model's performance on the test dataset.

```bash
python test_model.py```

This script provides a detailed analysis of:
-   Class distribution in the dataset.
-   Overall and per-class model performance.
-   Temporal prediction analysis.

### 3. Real-Time Inference

The inference script can run in three different modes to predict actions from various sources.

**From a Webcam:**
```bash
python inference.py --mode camera
```

**From a Video File:**
```bash
python inference.py --mode video --source video.mp4
```

**From a Directory of Screenshots:**
```bash
python inference.py --mode screenshots --screenshot_dir assets/data/raw
```

**Runtime Controls:**
-   Press `q` to quit the inference process.
-   Press `s` to save the current frame.

## 📊 Results and Analysis

The system is designed to automatically generate key performance indicators and visualizations.

1.  **Training Graphs:**
    -   Plots for training and validation loss and accuracy over epochs.
    -   A confusion matrix to visualize class-wise prediction accuracy.

2.  **Data Analysis:**
    -   Histograms showing the distribution of action classes in the dataset.
    -   Detailed reports on precision, recall, and F1-score for each class.

3.  **Temporal Predictions:**
    -   Visualization of predicted action sequences over time.
    -   Confidence scores for each prediction.

## 🔧 Advanced Configuration

### Architecture Modification

You can customize the model architecture directly within `train_model.py`.

```python
class Config:
    SEQUENCE_LENGTH = 10      # Length of input sequences
    FEATURE_DIM = 256         # Dimensionality of CNN features
    HIDDEN_SIZE = 128         # Hidden state size of the GRU
    DROPOUT_RATE = 0.3        # Dropout for regularization
```

### Change the CNN Backbone

To use a different pre-trained model for feature extraction, modify the `MCOCActionPredictor` class.

```python
# In MCOCActionPredictor.__init__()
self.cnn = models.efficientnet_b0(pretrained=True)  # Example: Replace MobileNetV2
```

### Use a Temporal Convolutional Network (TCN)

For a different approach to sequence modeling, you can replace the GRU with a TCN.

```python
# Replace the GRU layer with a TCN implementation
self.tcn = TemporalConvNet(...)
```

## 🎮 Bot Integration

The trained model can be seamlessly integrated into a larger bot framework.

```python
from scripts.inference import MCOCPredictor

# Load the trained model
predictor = MCOCPredictor("models/best_model.pth")

# Predict action from a single frame
action, confidence = predictor.predict_action(frame)
```

## 📈 Typical Performance

With a well-balanced dataset, the model can achieve:
-   **Accuracy**: 85-95%
-   **Inference FPS**: 4-5 frames per second
-   **Latency**: 200-250ms per prediction

## 🛠️ Troubleshooting

### Common Issues

1.  **CUDA out of memory:**
    -   Decrease the `BATCH_SIZE` during training.
    -   Reduce the `FEATURE_DIM` to lower memory consumption.

2.  **Overfitting:**
    -   Increase the `DROPOUT_RATE` in the model configuration.
    -   Implement data augmentation techniques to diversify the training data.

3.  **Poor Performance:**
    -   Ensure that the training data has a balanced class distribution.
    -   Experiment with a longer `SEQUENCE_LENGTH` to provide more context.
    -   Increase the size of the training dataset.

### Debugging Commands

```bash
# Run a quick test on a small subset of data
python test_model.py

# Verify the class distribution of your dataset
python -c "from scripts.test_model import analyze_data_distribution; analyze_data_distribution()"
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps to contribute:

1.  Fork the project.
2.  Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## 🙏 Acknowledgments

-   PyTorch for the powerful deep learning framework.
-   OpenCV for the essential computer vision tools.
-   The MCOC community for providing valuable training data.
