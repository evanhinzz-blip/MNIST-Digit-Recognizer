
**what this is:**
This is a Convolutional Neural Network (CNN) I built using TensorFlow to recognize handwritten numbers from the MNIST dataset(digits 0–9).
The model ended up getting around 99% accuracy, which was cool and honestly better than I expected when I first started.

**why I made it:**
I wanted to actually understand how neural networks “see” images instead of just following tutorials. This project helped me learn how convolution layers pick up on patterns like edges and shapes, and how those features get turned into predictions.

**data:**
- MNIST dataset  
- 60,000 training images  
- 10,000 test images  
- grayscale, 28×28 pixels

**how the model works:**
The CNN includes:
- convolution layers to extract features  
- max pooling layers to reduce image size  
- dense layers for classification  
- a softmax output to predict which digit it is  

I played around with the architecture and number of epochs to improve accuracy and avoid overfitting.

**results:**  
- test accuracy: ~99%  
- most errors happened with messy or ambiguous handwriting that i couldnt even tell what it was

**what I learned:**
- how CNNs actually process images  
- why architecture choices matter  
- how changing small things (like epochs or layer size) can affect results  
- that debugging ML models takes patience...
- This model only works well on clean MNIST-style images.

**tools used:**
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  


This was one of my first machine learning projects and helped set the foundation for applying deep learning to more meaningful, real-world problems later on.
I’m interested in using machine learning for social impact, and this project was a starting point for building those skills.

