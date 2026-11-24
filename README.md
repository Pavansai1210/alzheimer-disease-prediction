# REMDEx – AI Medical Assistant

REMDEx is an AI-powered medical assistant web application developed using Python and Flask. It provides basic medical information through a chatbot and includes an image classification feature for Alzheimer's disease prediction using brain MRI images.

## Features

- **AI Medical Chatbot:** Provides general information about diseases, symptoms, and precautions.
- **Alzheimer's Disease Prediction:** Classifies brain MRI images into four categories:
  - Mild Demented
  - Moderate Demented
  - Non-Demented
  - Very Mild Demented
- **User Authentication:** Registration and login functionality.
- **Web Interface:** User-friendly interface built with HTML, CSS, JavaScript, and Bootstrap.
- **Database:** SQLite database for user information.

## Technologies Used

- Python
- Flask
- TensorFlow
- Scikit-learn
- NLTK
- HTML, CSS, JavaScript
- Bootstrap
- SQLite
- Git and GitHub

## Project Structure

```text
REMDEx/
├── app.py
├── app1.py
├── bot.py
├── botserver.py
├── dataset/
├── templates/
├── static/
├── retrained_graph.pb
├── retrained_labels.txt
├── requirements.txt
└── README.md
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/Pavansai1210/REMDEx-AI-Medical-Assistant.git
```

2. Navigate to the project directory:

```bash
cd REMDEx-AI-Medical-Assistant
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Run the application:

```bash
python app1.py
```

7. Open your browser:

```text
http://127.0.0.1:5000
```

## Model Information

The project uses a TensorFlow-trained image classification model to classify brain MRI images into four Alzheimer's-related categories.

## Disclaimer

This project is developed for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice or diagnosis.

## Author

**Pavan Sai**

GitHub: https://github.com/Pavansai1210

LinkedIn: https://www.linkedin.com/in/bangaru-pavan-sai-ba784b27a/