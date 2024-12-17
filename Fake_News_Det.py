from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger le modèle préentraîné et le fichier vectorizer
loaded_model = pickle.load(open('model.pkl', 'rb'))
tfvect = pickle.load(open('vectorizer.pkl', 'rb'))  # Assurez-vous d'avoir sauvegardé le vectorizer

# Fonction de détection des fake news
def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)  # Transformer le texte en vecteurs
    prediction = loaded_model.predict(vectorized_input_data)  # Prédiction avec le modèle
    return prediction[0]  # Retourner la prédiction (0 ou 1)

# Route principale
@app.route('/')
def home():
    return render_template('index.html')  # Rendre la page d'accueil

# Route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Récupérer le texte saisi par l'utilisateur
            message = request.form['message']
            
            # Faire la prédiction
            pred = fake_news_det(message)
            result = 'Vrai' if pred == 1 else 'Fake'  # Interprétation du résultat
            
            return render_template('index.html', prediction=result)
        except Exception as e:
            return render_template('index.html', prediction=f"Erreur: {str(e)}")
    else:
        return render_template('index.html', prediction="Quelque chose a mal tourné.")

# Lancement de l'application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
