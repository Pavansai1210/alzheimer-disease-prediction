# REMDEx – AI Medical Assistant

REMDEx is an AI-powered medical assistant web application developed using Python and Flask. It combines a medical chatbot with an Alzheimer's disease prediction system that analyzes brain MRI images using a deep-learning image classification model.

The application also provides user registration, login, and a web-based interface for accessing its features.

## 🚀 Features

### 1. AI Medical Chatbot

* Provides general medical information about diseases and symptoms.
* Offers basic health information and precautions.
* Interactive chatbot interface for user queries.

### 2. Alzheimer's Disease Prediction

* Analyzes brain MRI images using a trained TensorFlow model.
* Classifies images into four categories:

  * Mild Demented
  * Moderate Demented
  * Non-Demented
  * Very Mild Demented

### 3. User Authentication

* User registration and login.
* Password hashing for user account security.
* SQLite database integration.

### 4. Web Interface

* User-friendly interface.
* Built using HTML, CSS, JavaScript, and Bootstrap.
* Separate sections for chatbot and MRI prediction.

## 🛠️ Technologies Used

| Category                         | Technologies                            |
| -------------------------------- | --------------------------------------- |
| Programming Language             | Python                                  |
| Backend                          | Flask                                   |
| Machine Learning / Deep Learning | TensorFlow, Scikit-learn                |
| Natural Language Processing      | NLTK                                    |
| Frontend                         | HTML, CSS, JavaScript, Bootstrap        |
| Database                         | SQLite                                  |
| OCR / Image Processing           | As applicable to project implementation |
| Version Control                  | Git, GitHub                             |

## 📂 Project Structure

```text
alzheimer-disease-prediction/
│
├── app.py
├── app1.py
├── bot.py
├── botserver.py
│
├── dataset/
├── templates/
├── static/
│
├── retrained_graph.pb
├── retrained_labels.txt
├── requirements.txt
├── run_project.bat
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Pavansai1210/alzheimer-disease-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd alzheimer-disease-prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app1.py
```

### 7. Open in Browser

```text
http://127.0.0.1:8000
```

## ▶️ Quick Start on Windows

The project includes a `run_project.bat` launcher that starts the Flask application and ngrok tunnel in separate command windows.

Double-click `run_project.bat` to launch the application.

**Note:** ngrok must be installed and configured on your system. The launcher uses the configured ngrok endpoint.

## 🧠 Model Information

The project uses a TensorFlow-trained image classification model to analyze brain MRI images and classify them into four Alzheimer's-related categories.

The model files are:

* `retrained_graph.pb`
* `retrained_labels.txt`

## 🔮 Future Enhancements

* Improve chatbot responses using advanced NLP and LLM integration.
* Enhance MRI image classification performance.
* Add more medical information and disease categories.
* Improve the user interface and overall application experience.

## ⚠️ Disclaimer

This project is developed for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## 👨‍💻 Author

**Pavan Sai**

* GitHub: https://github.com/Pavansai1210
* LinkedIn: https://www.linkedin.com/in/bangaru-pavan-sai-ba784b27a/
