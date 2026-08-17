import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from bot import MedicalChatbot, get_bot

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'medical-chatbot-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Database
db = SQLAlchemy(app)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize the chatbot
bot = get_bot()

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

# Pre-loading of AI model removed to prevent MemoryError during startup.
# The model will be initialized lazily on the first prediction request.

def load_image(image):
    # Lazy import to save memory during startup
    import label_image
    text = label_image.main(image)
    return text

# Routes
@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('register'))

        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('Email already registered!', 'error')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password!', 'error')
    
    return render_template('login.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Password reset link has been sent to your email!', 'success')
        else:
            flash('Email address not found!', 'error')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    return render_template("index1.html")

@app.route('/index')
@login_required
def index():
    return render_template('index.html')

@app.route('/chart')
@login_required
def chart():
    return render_template('chart.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        
        result = load_image(file_path)
        result = result.title()
        
        d = {
            "brownspot":" → You are Normal. Keep Calm and Make Healthy Life",
            'Mild Demented':" → You are Normal. Keep Calm and Make Healthy Life",
            "Moderate Demented":" → Neurodegenerative diseases... [Progressive Message]",
            "Non Demented":" → The most common neurodegenerative... [Info Message]",
            "Verymild Demented":" → Neurodegenerative diseases... [Clinical Message]"
        }
        
        result = result + d.get(result, "")
        os.remove(file_path)
        return result
    return ""

@app.route("/get")
@login_required
def get_bot_response():
    userText = request.args.get('msg')
    exit_list = ['exit', 'see you later', 'bye', 'quit', 'break']
    if userText.lower() in exit_list:
        return "Bye..Take Care..Chat with you later!!"
    else:
        return str(bot.chat(userText))

@app.errorhandler(500)
def internal_error(error):
    return "500 error", 500

@app.errorhandler(404)
def not_found(error):
    return "404 error", 404

if __name__ == '__main__':
    # Disabling the reloader can save memory by avoiding the second process
    app.run(debug=True, use_reloader=False)