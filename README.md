# 👕👟 Fashion Forward: Classifying Apparel with Deep Learning 🛍️✨

This project dives into the exciting world of **image classification** using **Convolutional Neural Networks (CNNs)**. To accurately classify fashion items from the popular **Fashion MNIST dataset**. Whether you're a machine learning enthusiast or just curious about how AI "sees" clothes, you'll find a practical demonstration here of building and training a CNN to recognize different types of apparel. Get ready to explore the code and see a deep learning model in action! 🚀

## 1. Dataset

* **Source:** The Fashion MNIST dataset is a well-known benchmark dataset available directly through the `tensorflow.keras.datasets` module. It was created by Zalando as a drop-in replacement for the original MNIST dataset.
    * **Official Documentation:** [https://www.tensorflow.org/datasets/catalog/fashion_mnist](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
* **Training Data:** 60,000 grayscale images. Each image is 28x28 pixels.
* **Testing Data:** 10,000 grayscale images. Each image is 28x28 pixels.
* **Classes:** The dataset consists of 10 distinct clothing categories:
    * 0: T-shirt/top
    * 1: Trouser
    * 2: Pullover
    * 3: Dress
    * 4: Coat
    * 5: Sandal
    * 6: Shirt
    * 7: Sneaker
    * 8: Bag
    * 9: Ankle boot

## 2. Methodology

### Model Architecture (CNN)

A simple Convolutional Neural Network (CNN) architecture was employed for image classification. The model consists of:
* Three `Conv2D` layers with ReLU activation, followed by `MaxPooling2D` layers for feature extraction and dimensionality reduction.
* A `Flatten` layer to convert the 2D feature maps into a 1D vector.
* Two `Dense` (fully connected) layers, with the final `Dense` layer using a `softmax` activation function to output probabilities for each of the 10 classes.

### Approach

1.  **Data Loading:** The Fashion MNIST dataset was loaded directly using `keras.datasets.fashion_mnist.load_data()`.
2.  **Preprocessing:**
    * Image pixel values were normalized from the range [0, 255] to [0, 1] by dividing by 255.0.
    * Images were reshaped to `(height, width, channels)` i.e., `(28, 28, 1)` to be compatible with the CNN input layer.
3.  **Model Compilation:** The model was compiled using:
    * **Optimizer:** `adam`
    * **Loss Function:** `SparseCategoricalCrossentropy` (suitable for integer labels)
    * **Metrics:** `accuracy`
4.  **Training:** The model was trained for **10 epochs**.
5.  **Evaluation:** The trained model's performance was evaluated on the unseen test set to determine its generalization capability.

## 3. Technologies

* **Python 3**
* **TensorFlow 2.x** (specifically `tf.keras` for building and training the CNN)
* **Numpy** (for numerical operations)
* **Matplotlib** (for plotting and visualization)
* **Google Colab** (for cloud-based execution with GPU acceleration)

---

**Note:** The Google Colab notebook used for this project can be found [here](link_to_your_colab_notebook_on_github).
