# CSCI 5448 Research Project

## Supplementary Code: Neural Network Implementations in PyTorch and JAX

This repo contains minimal neural network examples used as supporting artifacts for a research project in CSCI 5448. The project investigates how modern machine learning frameworks (PyTorch, JAX, and Equinox) use classic software design patterns within their training workflows. 
These code samples show how similar neural network structures are expressed in two different frameworks, one object oriented (PyTorch) and one functional (JAX).

---

## How to Run the Examples

### **PyTorch Version**
To install PyTorch via pip, use the following command, depending on your Python version:
```# Python 3.x 
pip3 install torch torchvision
```

Run:
```bash
python3 pytorch_mlp_example.py
```

### **JAX Version**     
To install JAX via pip, use the following command, depending on your Python version:
```# Python 3.x 
pip install -U jax
```
Run:
```bash
python3 jax_mlp_example.py
```

### **Equinox Version**     
To install Equinox via pip, use the following command, depending on your Python version:
```# Python 3.x 
pip install equinox
```
This can be run locally but you should have a system with a GPU. You can run this on Google Colab with a TPU for better results in a Jupyter Notebook.
Run:
```bash
python3 equinox_nn_example.py
```

## Dependencies
- Python 3.x
- PyTorch
- JAX
- Equinox
