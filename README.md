
# Sign Language Translation using CNN and LSTM

This project implements a real-time sign language translation system that captures live video, processes it frame-by-frame using a Convolutional Neural Network (CNN), and leverages a Long Short-Term Memory (LSTM) network for temporal analysis. The system translates hand signs into corresponding letters of the alphabet (A-Z) using the **Sign Language MNIST** dataset from Kaggle.

## Dataset

This project uses the **Sign Language MNIST** dataset from Kaggle for training the CNN model. The dataset consists of images of hand signs representing letters from A to Z.

- **Dataset Link**: [Sign Language MNIST on Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

## Project Overview

### Key Components:

1. **CNN Model**: Trained to extract features from individual frames of hand signs.
2. **LSTM Model**: Used for temporal sequence analysis over a series of frames to provide context for real-time predictions.
3. **Real-time Video Processing**: Uses OpenCV to capture video input and predict hand signs continuously.
4. **Translation to Letters**: Each prediction from the LSTM is mapped to a corresponding letter (A-Z).

### Technologies Used:
- **TensorFlow/Keras**: For building and training the CNN and LSTM models.
- **OpenCV**: For capturing and processing video in real-time.
- **NumPy**: For data manipulation.
- **Python**: The core programming language used for the implementation.

### System Architecture:
- **CNN**: Processes 28x28 grayscale images of hand signs and extracts features.
- **LSTM**: Takes a sequence of these features (from multiple frames) and predicts the corresponding letter.
- **Real-Time Translation**: Maps predicted classes (0-25) to letters (A-Z) and prints the result.

## Installation

### Prerequisites

- **Python 3.6+**
- **TensorFlow/Keras**
- **OpenCV**
- **NumPy**
- **A compatible webcam for real-time video capture**

### Steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sign-language-translation.git
   cd sign-language-translation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Kaggle dataset and place it in the `data/` directory:
   - Go to [Sign Language MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist).
   - Download and extract the dataset into the `data/` folder.

4. Train the CNN model on the dataset using the provided script:
   ```bash
   python train_cnn.py
   ```

5. Run the real-time sign language translation system:
   ```bash
   python sign_language_translation.py
   ```

## How It Works

1. The system captures video input using OpenCV.
2. Each frame is processed by the pre-trained CNN model to extract features.
3. These features are stored in a frame queue.
4. Once 10 frames are collected, the LSTM model uses the sequence of frames to predict the hand sign.
5. The predicted hand sign is translated into a letter (A-Z), which is printed as output.

## Model Architecture

### CNN Model:
- **Input**: 28x28 grayscale image (hand sign)
- **Layers**: 
  - 3 Convolutional layers
  - MaxPooling layers
  - Fully Connected layer
  - Output: Softmax layer with 25 units (for A-Z prediction)

### LSTM Model:
- **Input**: Sequence of 10 frames, each represented by 256 extracted features.
- **Layers**:
  - LSTM with 64 units (returning sequences)
  - LSTM with 128 units
  - Fully Connected layer with softmax for final classification

## Example Output

When the system predicts hand signs, it outputs the corresponding letter predictions in real time:

```plaintext
a b c d e f g h i j ...
```

The letters are continuously predicted as hand signs are captured through the webcam.

## Usage

1. Ensure your webcam is connected.
2. Run the sign language translation system using:
   ```bash
   python sign_language_translation.py
   ```
3. Watch the predictions print in real-time, as you show hand signs to the webcam.

## Model Training

The CNN model was trained using the **Sign Language MNIST** dataset, which contains hand signs corresponding to letters A-Z. After training, the model was saved for use in real-time translation.

## Future Enhancements

- Extend the system to recognize more complex signs or even words.
- Add a graphical interface for better user experience.
- Improve the accuracy by fine-tuning the LSTM for real-time performance.
