# Trading Pattern Identifier – AI-Based Chart Pattern Detection

## Overview

This project implements an AI-driven chart pattern identification system using TensorFlow and Keras.
It uses a custom Convolutional Neural Network (CNN) model to classify stock market charts into categories such as Head & Shoulders, Double Top, and No Pattern.
A Flask-based web interface allows users to upload chart images and receive predictions instantly.

## Features

* Custom-trained CNN for pattern detection
* TensorFlow + Keras deep learning pipeline
* Flask web interface for real-time predictions
* Image preprocessing using Keras utilities
* Training enhancement using EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint
* Clean architecture suited for research, learning, and portfolio use
* Supports PNG and JPG chart uploads

## Tech Stack

* Python
* TensorFlow / Keras
* Flask
* NumPy
* Pillow (PIL)

## Project Structure

```
Trading-Pattern-Identifier/
│
├── app.py                  # Flask application  
├── templates/              # HTML templates  
├── static/                 # Style and assets  
├── model/                  # Model directory (excluded from Git)  
├── training/               # Training scripts  
├── requirements.txt        # Dependency list  
├── .gitignore              # Git ignore rules  
└── README.md               # Documentation  
```

## Model Download

The trained `.h5` model is not included in the repository due to GitHub size limits.
Download the model manually from the link below:

**Direct Download:**
[https://drive.google.com/uc?export=download&id=1UrLsYanMs5Iell9jgM8R3J-MQ3AoQZ3P](https://drive.google.com/uc?export=download&id=1UrLsYanMs5Iell9jgM8R3J-MQ3AoQZ3P)

Place the downloaded file inside:

```
model/pattern_model.h5
```

## Installation & Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Flask server

```bash
python app.py
```

### 3. Access the application

Open your browser and navigate to:

```
http://localhost:5000
```

## Training

Training scripts are included under the `training/` directory.
These scripts prepare the dataset, apply augmentation, define the CNN architecture, and train the model using:

* ImageDataGenerator
* EarlyStopping
* ModelCheckpoint
* ReduceLROnPlateau

Run training with:

```bash
python training/train.py
```

## Use Cases

* Technical analysis automation
* Financial market research
* Pattern recognition systems
* Academic projects and BCA major/minor submissions
* Portfolio showcase for Data/Research Analyst roles

## License

This project is available for academic and personal learning use.

By:
Saloni kumari singh 
BCA'26
Sarala birla university 

 
Linkdin: https://www.linkedin.com/in/saloni-singh1329/
Github: https://github.com/inolas-bit


