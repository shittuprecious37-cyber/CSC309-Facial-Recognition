# Facial Recognition with SVM Classification

A Python-based facial recognition project developed as part of my Computer Science coursework. The project uses image processing and supervised machine learning to identify individuals from facial images.

**Model accuracy: 97.5% (39 out of 40 test images correctly classified).**

## Project Overview

| Component | Details |
|---|---|
| Task | Multi-class facial identification |
| Dataset | Olivetti Faces dataset |
| Total images | 200 |
| Individuals | 20 |
| Training set | 160 images |
| Testing set | 40 images |
| Classifier | Support Vector Machine (SVM) |
| Libraries | Python, OpenCV, NumPy, scikit-learn |

## Technologies Used

- Python
- OpenCV
- NumPy
- scikit-learn
- Support Vector Machine (SVM)

## Project Structure

| File | Purpose |
|---|---|
| `data.py` | Dataset loading and data handling |
| `process.py` | Image preprocessing |
| `model.py` | SVM model development and training |
| `test.py` | Model evaluation and testing |
| `main.py` | Main project execution |
| `processed_data.npz` | Saved processed dataset |
| `test_data.npz` | Saved test dataset |
| `face_model.pkl` | Saved trained model |
| `label_encoder.pkl` | Saved label encoder |

*File descriptions are based on their names and should be checked against the implementation.*

## Model Performance

The trained SVM classifier achieved:

**97.5% test accuracy — 39 correct predictions out of 40 test images.**

The project also involved examining classification performance and identifying prediction errors.

## Limitations

- The test set contains only 40 images, so one incorrect prediction changes the accuracy by 2.5 percentage points.
- The dataset is relatively small.
- Results on a controlled dataset may not represent performance under different lighting, facial angles and image conditions.

## Possible Improvements

- Evaluate the model using cross-validation.
- Experiment with different SVM kernels and parameters.
- Test with larger and more diverse facial image datasets.
- Improve image preprocessing and model evaluation.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/shittuprecious37-cyber/CSC309-Facial-Recognition.git
cd CSC309-Facial-Recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

The project includes `main.py` as its main execution file. Confirm that it runs the complete workflow before using:

```bash
python main.py
```

If the scripts must be executed separately, follow the execution order defined in the source code.

## Author

**Precious Shittu**

Computer Science Undergraduate  
Kwara State University (KWASU), Nigeria

[LinkedIn Profile](https://linkedin.com/in/precious-shittu-21)
