# bot.py - Complete Medical Chatbot with backward compatibility

import nltk
import warnings
import numpy as np
import random
import string
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DiseaseInfo:
    """Data class to store disease information"""
    name: str
    symptoms: List[str]
    medicines: List[str]
    precautions: List[str]
    corpus_file: str


class MedicalChatbot:
    """A medical information chatbot with multiple disease knowledge bases"""
    
    def __init__(self, corpus_dir: str = "."):
        """
        Initialize the chatbot with disease data and NLTK tools
        
        Args:
            corpus_dir: Directory containing corpus text files
        """
        self.corpus_dir = Path(corpus_dir)
        self.lemmer = nltk.stem.WordNetLemmatizer()
        self.remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
        
        # Initialize dictionaries for storing corpus data
        self.corpus_data = {}
        self.sent_tokens_dict = {}
        self.word_tokens_dict = {}
        
        # Initialize disease configurations
        self.diseases = self._init_disease_configs()
        
        # Load all corpus files
        self._load_all_corpora()
        
        # Conversation responses
        self.greeting_inputs = {"hello", "hi", "hiii", "hii", "greetings", "sup", "what's up", "hey"}
        self.greeting_responses = [
            "Hi! I'm Remdex. I can provide information about various diseases, symptoms, medicines, and precautions. How can I help you today?",
            "Hello! I'm here to assist with your medical queries. You can ask me about symptoms, treatments, or precautions for specific conditions.",
            "Greetings! I'm your medical assistant. I have information on many diseases including Alzheimer's. What would you like to know?"
        ]
        
        self.introduce_responses = [
            "My name is Remdex.", 
            "I'm Remdex, your medical assistant.",
            "My name is Remdex and I'm here to help with your queries.",
            "I'm Remdex :) How can I assist you today?"
        ]
        
        self.symptom_responses = [
            "I'm concerned to hear that.",
            "That sounds uncomfortable. Let's look into it.",
            "I'm sorry you're feeling this way. It's important to be cautious.",
            "Thank you for sharing. Please monitor these symptoms closely."
        ]
        
        # Basic FAQs
        self.faqs = {
            "what is m+ store": "M+ Store is an Online Medical Store. We supply medicines at your doorstep. You can order medicines on our website by uploading a proper prescription.",
            "from where you collect medicine": "We collect generic medicines from authentic sellers and Pradhan Mantri Bhartiya Jan Aushadhi Pariyojana Kendra, and deliver them to your doorstep at discounted prices.",
            "how much you charge": "We supply generic medicines at discount prices. Please visit our website for detailed pricing information.",
            "what is generic medicine": "Generic medicines are bioequivalent to brand-name drugs but cost less. Once a drug's patent expires, other companies can produce generic versions that meet the same quality standards."
        }
        
        # Symptom keywords
        self.symptom_keywords = {
            "fever", "headache", "vomitting", "sore throat", "cough", 
            "breathing problem", "pain", "itching"
        }
    
    def _init_disease_configs(self) -> Dict[str, DiseaseInfo]:
        """Initialize disease configurations with corpus file mappings"""
        disease_configs = {
            "diabetes": DiseaseInfo(
                name="Diabetes",
                symptoms=["increased thirst", "frequent urination", "hunger", "fatigue", "blurred vision"],
                medicines=["Metformin", "Insulin", "Sulfonylureas", "GLP-1 agonists"],
                precautions=["Monitor blood sugar", "Healthy diet", "Regular exercise", "Foot care"],
                corpus_file="corpus1.txt"
            ),
            "obesity": DiseaseInfo(
                name="Obesity",
                symptoms=["excess body fat", "shortness of breath", "snoring", "fatigue"],
                medicines=["Orlistat", "Phentermine-topiramate", "Bupropion-naltrexone"],
                precautions=["Balanced diet", "Regular exercise", "Adequate sleep", "Stress management"],
                corpus_file="corpus2.txt"
            ),
            "high blood pressure": DiseaseInfo(
                name="High Blood Pressure",
                symptoms=["headaches", "shortness of breath", "nosebleeds", "dizziness"],
                medicines=["Lisinopril", "Amlodipine", "Losartan", "Hydrochlorothiazide"],
                precautions=["Low sodium diet", "Regular exercise", "Limit alcohol", "Stress management"],
                corpus_file="corpus3.txt"
            ),
            "low blood pressure": DiseaseInfo(
                name="Low Blood Pressure",
                symptoms=["dizziness", "fainting", "blurred vision", "nausea", "fatigue"],
                medicines=["Fludrocortisone", "Midodrine", "Increased salt intake"],
                precautions=["Stay hydrated", "Stand up slowly", "Eat small meals", "Avoid alcohol"],
                corpus_file="corpus4.txt"
            ),
            "liver": DiseaseInfo(
                name="Liver Disease",
                symptoms=["jaundice", "abdominal pain", "swelling", "dark urine", "fatigue"],
                medicines=["Antivirals", "Corticosteroids", "Ursodeoxycholic acid"],
                precautions=["Avoid alcohol", "Healthy diet", "Regular exercise", "Vaccinations"],
                corpus_file="corpus5.txt"
            ),
            "heart": DiseaseInfo(
                name="Heart Disease",
                symptoms=["chest pain", "shortness of breath", "pain in arms", "dizziness"],
                medicines=["Statins", "Beta-blockers", "ACE inhibitors", "Aspirin"],
                precautions=["Heart-healthy diet", "Regular exercise", "No smoking", "Stress management"],
                corpus_file="corpus6.txt"
            ),
            "migrane": DiseaseInfo(
                name="Migraine",
                symptoms=["throbbing headache", "nausea", "sensitivity to light", "visual disturbances"],
                medicines=["Triptans", "NSAIDs", "Beta-blockers", "CGRP antagonists"],
                precautions=["Identify triggers", "Regular sleep", "Stay hydrated", "Stress reduction"],
                corpus_file="corpus7.txt"
            ),
            "diarrhoea": DiseaseInfo(
                name="Diarrhea",
                symptoms=["loose stools", "abdominal cramps", "urgency", "nausea"],
                medicines=["Oral rehydration salts", "Loperamide", "Bismuth subsalicylate"],
                precautions=["Stay hydrated", "Eat bland foods", "Rest", "Good hand hygiene"],
                corpus_file="corpus8.txt"
            ),
            "depression": DiseaseInfo(
                name="Depression",
                symptoms=["sadness", "loss of interest", "fatigue", "sleep changes", "appetite changes"],
                medicines=["SSRIs", "SNRIs", "Bupropion", "Cognitive behavioral therapy"],
                precautions=["Regular therapy", "Medication adherence", "Social connection", "Exercise"],
                corpus_file="corpus9.txt"
            ),
            "cancer": DiseaseInfo(
                name="Cancer",
                symptoms=["unexplained weight loss", "fatigue", "pain", "skin changes", "persistent cough"],
                medicines=["Chemotherapy", "Immunotherapy", "Targeted therapy", "Radiation"],
                precautions=["Regular screenings", "Healthy lifestyle", "Avoid tobacco", "Sun protection"],
                corpus_file="corpus10.txt"
            ),
            "kidney": DiseaseInfo(
                name="Kidney Disease",
                symptoms=["fatigue", "swelling", "urination changes", "shortness of breath"],
                medicines=["ACE inhibitors", "Diuretics", "Phosphate binders", "Erythropoietin"],
                precautions=["Control blood pressure", "Manage diabetes", "Low protein diet", "Avoid NSAIDs"],
                corpus_file="corpus11.txt"
            ),
            "insomnia": DiseaseInfo(
                name="Insomnia",
                symptoms=["difficulty sleeping", "waking early", "daytime fatigue", "irritability"],
                medicines=["Sleep hygiene", "Melatonin", "Sedative-hypnotics", "Cognitive behavioral therapy"],
                precautions=["Consistent sleep schedule", "Limit screen time", "Avoid caffeine", "Relaxation techniques"],
                corpus_file="corpus12.txt"
            ),
            "allergy": DiseaseInfo(
                name="Allergy",
                symptoms=["sneezing", "itching", "rash", "watery eyes", "runny nose"],
                medicines=["Antihistamines", "Decongestants", "Corticosteroids", "Epinephrine"],
                precautions=["Avoid allergens", "Use air purifiers", "Regular cleaning", "Carry emergency meds"],
                corpus_file="corpus13.txt"
            ),
            "dengue": DiseaseInfo(
                name="Dengue",
                symptoms=["high fever", "severe headache", "joint pain", "rash", "bleeding"],
                medicines=["Acetaminophen", "Fluid replacement", "Platelet transfusion if needed"],
                precautions=["Mosquito prevention", "Rest", "Hydration", "Monitor symptoms"],
                corpus_file="corpus14.txt"
            ),
            "malaria": DiseaseInfo(
                name="Malaria",
                symptoms=["fever", "chills", "sweating", "headache", "nausea"],
                medicines=["Artemisinin-based therapy", "Chloroquine", "Primaquine"],
                precautions=["Mosquito prevention", "Antimalarial drugs for travel", "Prompt treatment"],
                corpus_file="corpus15.txt"
            ),
            "corona": DiseaseInfo(
                name="COVID-19",
                symptoms=["fever", "cough", "fatigue", "loss of taste/smell", "difficulty breathing"],
                medicines=["Antipyretics", "Cough suppressants", "Antiviral medications"],
                precautions=["Vaccination", "Mask wearing", "Hand hygiene", "Social distancing"],
                corpus_file="corpus16.txt"
            ),
            "influenza": DiseaseInfo(
                name="Influenza",
                symptoms=["fever", "cough", "sore throat", "muscle aches", "fatigue"],
                medicines=["Oseltamivir", "Antipyretics", "Cough medicine", "Rest"],
                precautions=["Annual vaccination", "Hand washing", "Avoid close contact", "Stay home when sick"],
                corpus_file="corpus17.txt"
            ),
            "pneumonia": DiseaseInfo(
                name="Pneumonia",
                symptoms=["cough with phlegm", "fever", "chills", "difficulty breathing", "chest pain"],
                medicines=["Antibiotics", "Antivirals", "Cough medicine", "Fever reducers"],
                precautions=["Vaccination", "Good hygiene", "No smoking", "Rest and hydration"],
                corpus_file="corpus18.txt"
            ),

            "asthma": DiseaseInfo(
                name="Asthma",
                symptoms=["wheezing", "shortness of breath", "chest tightness", "coughing"],
                medicines=["Albuterol", "Fluticasone", "Montelukast", "Ipratropium"],
                precautions=["Avoid triggers", "Use inhaler correctly", "Regular checkups", "Avoid smoking"],
                corpus_file="corpus19.txt"
            ),
            "osteoarthritis": DiseaseInfo(
                name="Osteoarthritis",
                symptoms=["joint pain", "stiffness", "tenderness", "loss of flexibility", "grating sensation"],
                medicines=["NSAIDs", "Acetaminophen", "Topical pain relievers", "Corticosteroid injections"],
                precautions=["Gentle exercise", "Weight management", "Heat/cold therapy", "Joint protection"],
                corpus_file="corpus20.txt"
            ),
            "tuberculosis": DiseaseInfo(
                name="Tuberculosis",
                symptoms=["chronic cough", "chest pain", "coughing blood", "fatigue", "night sweats"],
                medicines=["Isoniazid", "Rifampin", "Ethanbutol", "Pyrazinamide"],
                precautions=["Complete medication course", "Cover mouth when coughing", "Good ventilation", "BCG vaccination"],
                corpus_file="corpus21.txt"
            ),
            "swine flu": DiseaseInfo(
                name="Swine Flu",
                symptoms=["fever", "cough", "sore throat", "body aches", "fatigue"],
                medicines=["Oseltamivir", "Zanamivir", "Antipyretics"],
                precautions=["Vaccination", "Hand hygiene", "Avoid contact with sick", "Cover coughs/sneezes"],
                corpus_file="corpus22.txt"
            ),
            "hiv": DiseaseInfo(
                name="HIV/AIDS",
                symptoms=["fever", "fatigue", "swollen lymph nodes", "weight loss", "opportunistic infections"],
                medicines=["Antiretroviral therapy", "Prophylactic antibiotics"],
                precautions=["Safe sex practices", "Needle safety", "Regular testing", "Medication adherence"],
                corpus_file="corpus23.txt"
            ),
            "leukemia": DiseaseInfo(
                name="Leukemia",
                symptoms=["fatigue", "fever", "frequent infections", "easy bleeding", "bone pain"],
                medicines=["Chemotherapy", "Targeted therapy", "Immunotherapy", "Stem cell transplant"],
                precautions=["Infection prevention", "Regular monitoring", "Supportive care", "Clinical trials"],
                corpus_file="corpus24.txt"
            ),
            "alzheimer": DiseaseInfo(
                name="Alzheimer's Disease",
                symptoms=["Memory loss", "Confusion with time/place", "Difficulty with familiar tasks", "Language problems", "Poor judgment", "Personality changes"],
                medicines=["Donepezil (Aricept)", "Rivastigmine (Exelon)", "Galantamine (Razadyne)", "Memantine (Namenda)", "Lecanemab (Leqembi)"],
                precautions=["Regular physical exercise", "Heart-healthy diet (Mediterranean)", "Mental stimulation/puzzles", "Social engagement", "Quality sleep", "Cardiovascular health management"],
                corpus_file="corpus25.txt"
            ),
            "anemia": DiseaseInfo(
                name="Anemia",
                symptoms=["Fatigue", "Weakness", "Pale skin", "Irregular heartbeats", "Dizziness", "Headaches"],
                medicines=["Iron supplements", "Vitamin B12", "Folic acid", "Erythropoietin"],
                precautions=["Iron-rich diet", "Vitamin C", "Avoid tea/coffee with meals", "Regular blood tests"],
                corpus_file="corpus26.txt"
            ),
            "arthritis": DiseaseInfo(
                name="Arthritis",
                symptoms=["Joint pain", "Stiffness", "Swelling", "Redness", "Decreased range of motion"],
                medicines=["NSAIDs", "Analgesics", "Corticosteroids", "DMARDs"],
                precautions=["Healthy weight", "Low-impact exercise", "Hot/cold therapy", "Joint protection"],
                corpus_file="corpus27.txt"
            ),
            "bronchitis": DiseaseInfo(
                name="Bronchitis",
                symptoms=["Persistent cough", "Mucus production", "Fatigue", "Shortness of breath", "Chest discomfort"],
                medicines=["Cough suppressants", "Bronchodilators", "Antipyretics"],
                precautions=["Avoid smoking", "Use humidifier", "Stay hydrated", "Get rest"],
                corpus_file="corpus28.txt"
            ),
            "cataract": DiseaseInfo(
                name="Cataract",
                symptoms=["Clouded vision", "Night vision difficulty", "Sensitivity to light", "Halos around lights"],
                medicines=["Prescription glasses", "Surgery (Advanced stage)", "Anti-inflammatory drops"],
                precautions=["Regular eye exams", "Quit smoking", "UV protection", "Antioxidant-rich diet"],
                corpus_file="corpus29.txt"
            ),
            "chickenpox": DiseaseInfo(
                name="Chickenpox",
                symptoms=["Itchy red rash", "Blisters", "Fever", "Headache", "Loss of appetite"],
                medicines=["Acetaminophen", "Calamine lotion", "Antivirals (Acyclovir)"],
                precautions=["Avoid scratching", "Isolate until scabbed", "Cool baths", "Keep nails short"],
                corpus_file="corpus30.txt"
            ),
            "cholera": DiseaseInfo(
                name="Cholera",
                symptoms=["Severe watery diarrhea", "Vomiting", "Rapid heart rate", "Muscle cramps", "Dehydration"],
                medicines=["ORS", "IV fluids", "Antibiotics (Doxycycline)", "Zinc"],
                precautions=["Safe water drinking", "Hand hygiene", "Cooked food", "Sanitary waste disposal"],
                corpus_file="corpus31.txt"
            ),
            "common cold": DiseaseInfo(
                name="Common Cold",
                symptoms=["Runny nose", "Sore throat", "Cough", "Congestion", "Sneezing", "Malaise"],
                medicines=["Pain relievers", "Decongestants", "Cough syrups", "Antihistamines"],
                precautions=["Frequent hand washing", "Disinfect surfaces", "Stay hydrated", "Rest"],
                corpus_file="corpus32.txt"
            ),
            "constipation": DiseaseInfo(
                name="Constipation",
                symptoms=["Hard stools", "Straining", "Feeling of blockage", "Infrequent movements"],
                medicines=["Fiber supplements", "Laxatives", "Stool softeners"],
                precautions=["High-fiber diet", "Adequate hydration", "Regular exercise", "Routine schedule"],
                corpus_file="corpus33.txt"
            ),
            "epilepsy": DiseaseInfo(
                name="Epilepsy",
                symptoms=["Seizures", "Confusion", "Staring spells", "Jerking movements", "Loss of awareness"],
                medicines=["Antiepileptic drugs", "Valproate", "Lamotrigine", "Levetiracetam"],
                precautions=["Medication adherence", "Adequate sleep", "Stress management", "Safety awareness"],
                corpus_file="corpus34.txt"
            ),
            "gastritis": DiseaseInfo(
                name="Gastritis",
                symptoms=["Stomach burning", "Nausea", "Indigestion", "Loss of appetite", "Fullness"],
                medicines=["PPIs", "H2 blockers", "Antacids", "Antibiotics (H. pylori)"],
                precautions=["Small frequent meals", "Avoid spicy foods", "Limit alcohol", "Stress management"],
                corpus_file="corpus35.txt"
            ),
            "glaucoma": DiseaseInfo(
                name="Glaucoma",
                symptoms=["Peripheral vision loss", "Eye pain", "Blurred vision", "Halos around lights", "Nausea"],
                medicines=["Prostaglandin drops", "Beta blockers", "Oral medications"],
                precautions=["Regular eye exams", "Know family history", "Exercise safely", "Follow eye drop schedule"],
                corpus_file="corpus36.txt"
            ),
            "hepatitis": DiseaseInfo(
                name="Hepatitis",
                symptoms=["Jaundice", "Fatigue", "Dark urine", "Pale stool", "Abdominal pain"],
                medicines=["Antivirals", "Interferon", "Supportive care"],
                precautions=["Vaccination (A and B)", "Hand hygiene", "Avoid sharing razors", "Safe sex"],
                corpus_file="corpus37.txt"
            ),
            "jaundice": DiseaseInfo(
                name="Jaundice",
                symptoms=["Yellow skin/eyes", "Itchy skin", "Dark urine", "Pale stool", "Fatigue"],
                medicines=["Treat underlying cause", "Itch relief meds", "Phototherapy (Infants)"],
                precautions=["Avoid alcohol", "Hepatitis vaccination", "Healthy weight", "Safe needles"],
                corpus_file="corpus38.txt"
            ),
            "lupus": DiseaseInfo(
                name="Lupus",
                symptoms=["Butterfly rash", "Joint pain", "Fatigue", "Sun sensitivity", "Fever"],
                medicines=["NSAIDs", "Corticosteroids", "Hydroxychloroquine", "Immunosuppressants"],
                precautions=["Sun protection", "Regular exercise", "Quit smoking", "Stress management"],
                corpus_file="corpus39.txt"
            ),
            "measles": DiseaseInfo(
                name="Measles",
                symptoms=["High fever", "Blotchy rash", "Cough", "Runny nose", "White spots in mouth"],
                medicines=["Acetaminophen", "Vitamin A", "Supportive care"],
                precautions=["MMR vaccine", "Isolate infected person", "Hydration", "Rest"],
                corpus_file="corpus40.txt"
            ),
            "parkinson": DiseaseInfo(
                name="Parkinson's Disease",
                symptoms=["Tremors", "Slowed movement", "Rigid muscles", "Impaired posture", "Speech changes"],
                medicines=["Carbidopa-levodopa", "Dopamine agonists", "MAO B inhibitors"],
                precautions=["Regular aerobic exercise", "Physical therapy", "Fall prevention", "Healthy diet"],
                corpus_file="corpus41.txt"
            ),
            "psoriasis": DiseaseInfo(
                name="Psoriasis",
                symptoms=["Itchy scaly patches", "Nail pitting", "Dry cracked skin", "Soreness"],
                medicines=["Corticosteroids", "Vitamin D analogues", "Phototherapy", "Retinoids"],
                precautions=["Moisturize regularly", "Identify/avoid triggers", "Quit smoking", "Limit alcohol"],
                corpus_file="corpus42.txt"
            ),
            "ringworm": DiseaseInfo(
                name="Ringworm",
                symptoms=["Ring-shaped rash", "Itching", "Scaly skin", "Raised border"],
                medicines=["Antifungal creams", "Antifungal powders", "Oral antifungals"],
                precautions=["Keep skin dry", "Don't share towels", "Wash hands after pets", "Wear breathable shoes"],
                corpus_file="corpus43.txt"
            ),
            "scabies": DiseaseInfo(
                name="Scabies",
                symptoms=["Intense itching at night", "Thin wavy tunnels", "Pimple-like rash", "Sores from scratching"],
                medicines=["Permethrin cream", "Ivermectin", "Crotamiton", "Antihistamines"],
                precautions=["Wash bedding in hot water", "Treat whole family", "Avoid skin contact", "Vacuum home"],
                corpus_file="corpus44.txt"
            ),
            "scurvy": DiseaseInfo(
                name="Scurvy",
                symptoms=["Bleeding gums", "Weakness", "Bruising", "Poor wound healing", "Joint pain"],
                medicines=["Vitamin C supplements", "Ascorbic acid", "Dietary Vitamin C"],
                precautions=["Eat fresh fruits/veg", "Include citrus", "Avoid overcooking veg", "Daily Vitamin C intake"],
                corpus_file="corpus45.txt"
            ),
            "stroke": DiseaseInfo(
                name="Stroke",
                symptoms=["Face drooping", "Arm weakness", "Speech difficulty", "Sudden confusion", "Numbness"],
                medicines=["Clot-busting drugs (tPA)", "Blood thinners", "Statins", "BP control"],
                precautions=["Quit smoking", "Manage BP", "Healthy low-fat diet", "Regular exercise"],
                corpus_file="corpus46.txt"
            ),
            "typhoid": DiseaseInfo(
                name="Typhoid",
                symptoms=["Prolonged high fever", "Headache", "Stomach pain", "Diarrhea", "Fatigue"],
                medicines=["Antibiotics (Ciprofloxacin)", "Azithromycin", "Fluids"],
                precautions=["Vaccination", "Safe drinking water", "Hand washing", "Cooked food"],
                corpus_file="corpus47.txt"
            ),
            "ulcer": DiseaseInfo(
                name="Ulcer",
                symptoms=["Burning stomach pain", "Nausea", "Fullness/Bloating", "Heartburn"],
                medicines=["Antibiotics", "PPIs", "H2 blockers", "Antacids"],
                precautions=["Limit NSAIDs", "Avoid alcohol", "Quit smoking", "Manage stress"],
                corpus_file="corpus48.txt"
            ),
            "uti": DiseaseInfo(
                name="UTI",
                symptoms=["Burning urination", "Frequent urge", "Cloudy urine", "Pelvic pain"],
                medicines=["Antibiotics (Nitrofurantoin)", "Pain relief meds", "Antispasmodics"],
                precautions=["Drink plenty of water", "Wipe front-to-back", "Urinate after intercourse", "Avoid irritant products"],
                corpus_file="corpus49.txt"
            ),
            "vertigo": DiseaseInfo(
                name="Vertigo",
                symptoms=["Spinning sensation", "Nausea", "Balance issues", "Headache", "Sweating"],
                medicines=["Meclizine", "Diazepam", "Antihistamines", "Scopolamine"],
                precautions=["Move head slowly", "Sit down when dizzy", "Manage stress", "Epley maneuver"],
                corpus_file="corpus50.txt"
            ),
            "fever": DiseaseInfo(
                name="Fever",
                symptoms=["High body temperature", "Chills and shivering", "Sweating", "Headache", "Muscle aches", "Weakness"],
                medicines=["Paracetamol (Acetaminophen)", "Ibuprofen", "Naproxen", "Aspirin (adults only)"],
                precautions=["Drink plenty of fluids", "Get adequate rest", "Wear lightweight clothing", "Use a lukewarm compress", "Avoid aspirin in children"],
                corpus_file="corpus51.txt"
                ),
        }
        
        # Add alternative names
        self.disease_aliases = {
            "diabetes": ["diabetes", "sugar", "high blood sugar"],
            "obesity": ["obesity", "overweight"],
            "high blood pressure": ["high blood pressure", "hypertension"],
            "low blood pressure": ["low blood pressure", "hypotension"],
            "liver": ["liver", "liver disease"],
            "heart": ["heart", "heart disease"],
            "migrane": ["migrane", "migraine"],
            "diarrhoea": ["diarrhoea", "diarrhea"],
            "depression": ["depression", "anxiety"],
            "corona": ["corona", "covid", "covid19", "coronavirus"],
            "influenza": ["influenza", "flu"],
            "tuberculosis": ["tuberculosis", "tb"],
            "hiv": ["hiv", "aids"],
            "alzheimer": ["alzheimer", "alzheimers", "dementia", "memory loss", "forgetting"],
            "anemia": ["anemia", "low blood", "low hemoglobin"],
            "arthritis": ["arthritis", "joint pain", "bone pain"],
            "bronchitis": ["bronchitis"],
            "cataract": ["cataract", "cloudy vision", "eye lens"],
            "chickenpox": ["chickenpox", "varicella"],
            "cholera": ["cholera", "water diarrhea"],
            "common cold": ["cold", "flu symptoms", "sneezing", "cough"],
            "constipation": ["constipation", "hard stool"],
            "epilepsy": ["epilepsy", "seizure", "staring spells"],
            "gastritis": ["gastritis", "stomach inflammation"],
            "glaucoma": ["glaucoma", "eye pressure"],
            "hepatitis": ["hepatitis", "liver inflammation"],
            "jaundice": ["jaundice", "yellow skin", "yellow eyes"],
            "lupus": ["lupus", "butterfly rash"],
            "measles": ["measles", "blotchy rash"],
            "parkinson": ["parkinson", "parkinsons", "tremors", "shaking"],
            "psoriasis": ["psoriasis", "scaly skin"],
            "ringworm": ["ringworm", "fungal rash"],
            "scabies": ["scabies", "itchy rash at night"],
            "scurvy": ["scurvy", "bleeding gums"],
            "stroke": ["stroke", "brain attack", "paralysis"],
            "typhoid": ["typhoid", "enteric fever"],
            "ulcer": ["ulcer", "stomach sore"],
            "uti": ["uti", "urinary infection", "bladder infection"],
            "vertigo": ["vertigo", "spinning", "dizziness"],
            "fever": ["fever","high temperature","high body temperature","pyrexia","febrile","chills","shivering","feeling hot","body warmth","running a temperature"]

        }
        
        return disease_configs
    
    def _load_all_corpora(self):
        """Load all corpus files into memory"""
        for disease_name, disease_info in self.diseases.items():
            file_path = self.corpus_dir / disease_info.corpus_file
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    raw_text = f.read().lower()
                    sent_tokens = nltk.sent_tokenize(raw_text)
                    word_tokens = nltk.word_tokenize(raw_text)
                    
                    self.corpus_data[disease_name] = {
                        'raw': raw_text,
                        'sent_tokens': sent_tokens,
                        'word_tokens': word_tokens
                    }
                    
                    # Store in separate dictionaries for backward compatibility
                    self.sent_tokens_dict[disease_name] = sent_tokens
                    self.word_tokens_dict[disease_name] = word_tokens
                    
            except FileNotFoundError:
                print(f"Warning: Corpus file {file_path} not found. Using default information for {disease_name}.")
                # Use default sentences if file not found
                default_sents = [f"Information about {disease_info.name}. {disease_info.name} is a medical condition."]
                self.corpus_data[disease_name] = {
                    'raw': f"Information about {disease_name}.",
                    'sent_tokens': default_sents,
                    'word_tokens': []
                }
                self.sent_tokens_dict[disease_name] = default_sents
                self.word_tokens_dict[disease_name] = []
    
    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize a list of tokens"""
        return [self.lemmer.lemmatize(token) for token in tokens]
    
    def _normalize_text(self, text: str) -> List[str]:
        """Normalize text by lowercasing, removing punctuation, and lemmatizing"""
        text = text.lower().translate(self.remove_punct_dict)
        tokens = nltk.word_tokenize(text)
        return self._lemmatize_tokens(tokens)
    
    def _get_response_from_corpus(self, disease_name: str, user_input: str) -> str:
        """Generate response using TF-IDF similarity for a specific disease"""
        if disease_name not in self.corpus_data:
            return f"I'm sorry! I don't have information about {disease_name}"
        
        sent_tokens = self.corpus_data[disease_name]['sent_tokens'].copy()
        sent_tokens.append(user_input)
        
        try:
            tfidf_vec = TfidfVectorizer(tokenizer=self._normalize_text, stop_words='english')
            tfidf_matrix = tfidf_vec.fit_transform(sent_tokens)
            cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix)
            
            # Get the most similar sentence (excluding the user input itself)
            similar_indices = cosine_similarities.argsort()[0][-2]
            similarity_score = cosine_similarities[0][similar_indices]
            
            if similarity_score < 0.2:
                # If similarity is low, fallback to the first meaningful sentence (usually the overview)
                return sent_tokens[0] if len(sent_tokens) > 0 else "I'm sorry, I don't have enough details yet."
            
            return sent_tokens[similar_indices]
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I am sorry! I don't understand you. Please try asking about symptoms, medicines, or precautions."
    
    def _check_greeting(self, user_input: str) -> Optional[str]:
        """Check if input is a greeting and return response"""
        for word in user_input.split():
            if word in self.greeting_inputs:
                return random.choice(self.greeting_responses)
        return None
    
    def _check_introduction(self, user_input: str) -> Optional[str]:
        """Check if user is asking for bot's name"""
        name_queries = ["your name", "who are you", "introduce yourself", "your name"]
        if any(query in user_input for query in name_queries):
            return random.choice(self.introduce_responses)
        return None
    
    def _check_faq(self, user_input: str) -> Optional[str]:
        """Check if user asked an FAQ"""
        for question, answer in self.faqs.items():
            if question in user_input:
                return answer
        return None
    
    def _check_symptoms(self, user_input: str) -> Optional[str]:
        """Check if user mentioned symptoms"""
        mentioned_symptoms = []
        for symptom in self.symptom_keywords:
            if symptom in user_input:
                mentioned_symptoms.append(symptom)
        
        if mentioned_symptoms:
            response = "<strong>Remdex Clinical Insight:</strong><br>" + random.choice(self.symptom_responses) + "<br><br>"
            response += "This could be a significant symptom. <strong>Please consult a medical professional immediately.</strong>"
            return response
        return None
    
    def _identify_disease(self, user_input: str) -> Optional[str]:
        """Identify which disease the user is asking about"""
        # Direct match
        for disease in self.diseases.keys():
            if disease in user_input:
                return disease
        
        # Check aliases
        for disease, aliases in self.disease_aliases.items():
            if any(alias in user_input for alias in aliases):
                return disease
        
        return None
    
    def chat(self, user_input: str) -> str:
        """
        Main chat function to process user input and return response
        
        Args:
            user_input: The user's message
            
        Returns:
            Bot's response
        """
        user_input = user_input.lower().strip()
        
        # Check for exit commands
        exit_commands = ['exit', 'bye', 'quit', 'see you later', 'goodbye', 'break']
        if user_input in exit_commands:
            return "ROBO: Bye! Take care..Chat with you later!!"
        
        # Check for thanks
        if user_input in ['thanks', 'thank you']:
            return "You are welcome.."
        
        # Check greeting
        greeting_response = self._check_greeting(user_input)
        if greeting_response:
            return greeting_response
        
        
        # Check introduction
        intro_response = self._check_introduction(user_input)
        if intro_response:
            return intro_response

        # Check FAQ
        faq_response = self._check_faq(user_input)
        if faq_response:
            return faq_response

        # Help command
        if user_input in ["help", "what can you do", "commands", "menu"]:
            return "Your existing help response"

        # Check for specific disease FIRST
        disease = self._identify_disease(user_input)

        
        if disease:
            disease_info = self.diseases[disease]

            resp = f"<h3>{disease_info.name}</h3>"

            # Overview
            summary = self._get_response_from_corpus(disease, user_input)
            resp += f"<p><strong>Overview:</strong> {summary}</p>"

            # Symptoms
            resp += "<h4>Common Symptoms:</h4><ul>"
            for symptom in disease_info.symptoms:
                resp += f"<li>{symptom}</li>"
            resp += "</ul>"

            # Medicines
            resp += "<h4>Medicines:</h4><ul>"
            for medicine in disease_info.medicines:
                resp += f"<li>{medicine}</li>"
            resp += "</ul>"
            resp += "<p>Medication should only be taken after consulting a doctor.</p>"

            # Precautions
            resp += "<h4>Precautions:</h4><ul>"
            for precaution in disease_info.precautions:
                resp += f"<li>{precaution}</li>"
            resp += "</ul>"

            return resp


        # Check general symptoms AFTER disease detection
        symptom_response = self._check_symptoms(user_input)

        if symptom_response:
            return symptom_response

        
            # Overview from corpus
            summary = self._get_response_from_corpus(disease, user_input)
            resp += f"<p><strong>Overview:</strong> {summary}</p>"
            
            # Symptoms
            resp += "<strong>Key Symptoms:</strong><ul>"
            for s in disease_info.symptoms[:4]: # Show top 4
                resp += f"<li>{s}</li>"
            resp += "</ul>"
            
            # Medicines
            resp += "<strong>Treatment/Medicines:</strong><ul>"
            for m in disease_info.medicines[:3]: # Show top 3
                resp += f"<li>{m}</li>"
            resp += "</ul>"
            
            # Precautions
            resp += "<strong>Precautions & Care:</strong><ul>"
            for p in disease_info.precautions[:3]: # Show top 3
                resp += f"<li>{p}</li>"
            resp += "</ul>"
            
            resp += "<br><small><em>Disclaimer: This information is for educational purposes. Always consult a healthcare professional.</em></small>"
            return resp
                
            return self._get_response_from_corpus(disease, user_input)
        
        # Check for medicine-related queries
        if "medicine" in user_input or "medication" in user_input:
            return "<strong>Medical Safety Notice:</strong><br>Please take any medication <em>only</em> after consulting with a licensed physician. You can order prescription medicines from the M+ store by uploading a valid prescription."
        
        # Default response
        return "I am just a chatbot. Please consult a doctor for your query."


# Create a global instance for backward compatibility
_bot_instance = None

def get_bot():
    """Get or create the global bot instance"""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = MedicalChatbot(corpus_dir=".")
    return _bot_instance


# Backward compatibility functions
def chat(user_response):
    """Backward compatible chat function"""
    bot = get_bot()
    return bot.chat(user_response)


def response1(user_response):
    """Backward compatible response function for disease 1"""
    bot = get_bot()
    return bot._get_response_from_corpus("diabetes", user_response)


def response2(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("obesity", user_response)


def response3(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("high blood pressure", user_response)


def response4(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("low blood pressure", user_response)


def response5(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("liver", user_response)


def response6(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("heart", user_response)


def response7(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("migrane", user_response)


def response8(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("diarrhoea", user_response)


def response9(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("depression", user_response)


def response10(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("cancer", user_response)


def response11(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("kidney", user_response)


def response12(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("insomnia", user_response)


def response13(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("allergy", user_response)


def response14(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("dengue", user_response)


def response15(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("malaria", user_response)


def response16(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("corona", user_response)


def response17(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("influenza", user_response)


def response18(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("pneumonia", user_response)


def response19(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("asthma", user_response)


def response20(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("osteoarthritis", user_response)


def response21(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("tuberculosis", user_response)


def response22(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("swine flu", user_response)


def response23(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("hiv", user_response)


def response24(user_response):
    bot = get_bot()
    return bot._get_response_from_corpus("leukemia", user_response)


def greeting(sentence):
    """Backward compatible greeting function"""
    bot = get_bot()
    return bot._check_greeting(sentence)


def IntroduceMe(sentence):
    """Backward compatible introduction function"""
    bot = get_bot()
    return bot._check_introduction(sentence)


def findsymptom():
    """Backward compatible symptom response function"""
    bot = get_bot()
    return random.choice(bot.symptom_responses)


def basic(sentence):
    """Backward compatible FAQ function for basic questions"""
    bot = get_bot()
    return bot._check_faq(sentence)


# Make all functions available for import
__all__ = [
    'chat', 'greeting', 'IntroduceMe', 'findsymptom', 'basic',
    'response1', 'response2', 'response3', 'response4', 'response5',
    'response6', 'response7', 'response8', 'response9', 'response10',
    'response11', 'response12', 'response13', 'response14', 'response15',
    'response16', 'response17', 'response18', 'response19', 'response20',
    'response21', 'response22', 'response23', 'response24',
    'MedicalChatbot', 'get_bot'
]


# Example usage
if __name__ == "__main__":
    # Create chatbot instance
    bot = MedicalChatbot(corpus_dir=".")
    
    print("Medical Chatbot initialized. Type 'exit' to quit.")
    print("=" * 50)
    
    # Simple chat loop
    while True:
        user_input = input("You: ")
        response = bot.chat(user_input)
        print(f"Bot: {response}")
        
        if user_input.lower() in ['exit', 'bye', 'quit']:
            break