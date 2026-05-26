# TSYNET AI Orthodontic Growth Predictor

An AI-powered web application that predicts orthodontic conditions from dental X-ray images using Deep Learning and Convolutional Neural Networks (CNN).

---

## Live Demo

[Live Demo](https://ai-powered-orthodontic-growth-predictor.onrender.com/)

---

## Features

- AI-powered orthodontic prediction
- Dental X-ray image analysis
- Predicts:
  - Normal
  - Overbite
  - Underbite
- Real-time prediction system
- Flask web application
- Responsive UI using Tailwind CSS
- Deployed using Render

---

## Tech Stack

### Frontend
- HTML
- Tailwind CSS

### Backend
- Flask (Python)

### AI / Deep Learning
- TensorFlow
- Keras
- CNN (Convolutional Neural Network)

### Deployment
- Render
- GitHub

---

## Project Structure

```bash
AI-Powered-Orthodontic-Growth-Predictor/
│
├── models/
│   └── tsynet_ortho_predictor.h5
│
├── static/
│   └── uploads/
│
├── templates/
│   ├── home.html
│   └── dashboard.html
│
├── nd.py
├── train.py
├── requirements.txt
├── Procfile
├── .python-version
├── .gitignore
└── README.md
```

---

## Model Information

The project uses a CNN-based deep learning model trained on orthodontic X-ray images classified into:

- Normal
- Overbite
- Underbite

The model performs:
- Image preprocessing
- Feature extraction
- Orthodontic classification

using TensorFlow and Keras.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/AI-Powered-Orthodontic-Growth-Predictor.git
```

---

### Move Into Project Folder

```bash
cd AI-Powered-Orthodontic-Growth-Predictor
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

---

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
python nd.py
```

---

## Deployment

The application is deployed using Render.

### Procfile

```bash
web: gunicorn nd:app
```

### Python Version

```bash
3.10.13
```

---

## Future Enhancements

- Disease severity prediction
- Grad-CAM heatmaps
- PDF medical report generation
- Doctor login system
- Improved dataset and model accuracy

---

## Author

**Asritha Nalubala**


---

## License

This project is developed for educational and research purposes.
