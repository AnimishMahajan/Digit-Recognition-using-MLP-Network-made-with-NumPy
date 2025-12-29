# Digit Recognition using MLP Network made with NumPy
Handwritten Digit Recognition model using MLP network which was made only using Python and NumPy.

🚀 Overview

This project demonstrates the internal mechanics of deep learning by implementing all essential components manually. By avoiding high-level frameworks like TensorFlow or PyTorch.

The templates(for the webpage) were generated via Google Gemini.

🧠 Technical Features

* Pure NumPy: All matrix operations and transformations are handled via NumPy.

* Manual Backpropagation: Gradients are calculated using Gradient Descent.

* Vectorized Implementation: Efficient batch processing using matrix operations.

* Customizable Architecture: Easily adjust the number of hidden layers and neurons.

⚙️ How to Run

Once you have installed the repo and created the environment required to run, you can you the following commands,

```bash 
    $ python src\main.py
```
or

```bash 
    $ uvicorn src.main:app --host 127.0.0.1 --port 8000
```

✨ Special Thanks

Deepest thanks to [Dr. Pramod Kachare](https://in.linkedin.com/in/pramodhkachare) for his exceptional mentorship. This project would not have been possible without his excellent teaching skills which helped simplify the complex mathematical foundations of deep learning. His profound insight into neural network architectures particularly vectorized implementations and backpropagation logic was instrumental in guiding me to build this model from the ground up using only NumPy.
