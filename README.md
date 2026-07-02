# Digit Recognition using MLP Network made with NumPy
Handwritten Digit Recognition model using MLP network which was made only using Python and NumPy.

---

### 🚀 Overview

This project demonstrates the internal mechanics of deep learning by implementing all essential components manually. By avoiding high-level frameworks like TensorFlow or PyTorch.

The templates(for the webpage) were generated via Google Gemini.

---

### 🧠 Technical Features

* Pure NumPy: All matrix operations and transformations are handled via NumPy.

* Manual Backpropagation: Gradients are calculated using Gradient Descent.

* Vectorized Implementation: Efficient batch processing using matrix operations.

* Customizable Architecture: Easily adjust the number of hidden layers and neurons.

---

### 🪛 Tech Stack

**Core Language:** Python 3.10 (Can be used with newer versions as well)
\
\
**Libraries:**
* NumPy - for the NN
* FastAPI - for hosting the model
* jinja2
* uvicorn
* pydantic

---

### ⚙️ How to Run(Installation)

Once you have installed the repo, you can use the following commands,

* Installing required libraries
```bash
$ pip install - requirements.txt
```

* Hosting the model

```bash 
$ python src\main.py
```
or use this if you want to use a specific ip and port

```bash 
$ uvicorn src.main:app --host 127.0.0.1 --port 8000
```

**NOTE:** Python must be installed on your system.

---

### 🧩 Steps to use the library for yourself

A sample code to create your own network and using it is shown in the "src\test.py" file.

---

### 🗺️ Project Structure

```
├── models/
│   └── mnist/
│       ├── model_arch.npz
│       └── params.npz
├── src/
│   ├── templates/
│   │   └── index.html
│   ├── main.py
│   ├── model.py
│   └── nn_npy.py
├── Dockerfile.dockerfile
├── LICENSE
├── mlops.yaml
├── README.md
└── requirements.txt

```
---

### ˙◠˙ Known Isuues and Limitations

This Network does not use any optimizers yet causing slower training and poor results. The Early stopping feature has not been implemented correctly, I plan on fixing it later but it is not my top priority at the moment.

---
### ✨ Special Thanks

Deepest thanks to [Dr. Pramod Kachare](https://in.linkedin.com/in/pramodhkachare) for his exceptional mentorship. This project would not have been possible without his excellent teaching skills which helped simplify the complex mathematical foundations of deep learning. His profound insight into neural network architectures particularly vectorized implementations and backpropagation logic was instrumental in guiding me to build this model from the ground up using only NumPy.
