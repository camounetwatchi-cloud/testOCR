#!/usr/bin/env python3
"""
Interface web Flask pour l'outil OCR
"""

import os
import json
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, jsonify, send_from_directory
from ocr_tool import OCRProcessor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'images'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Créer les dossiers nécessaires
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

# Initialiser le processeur OCR
print("🔄 Initialisation du processeur OCR...")
ocr_processor = OCRProcessor(languages=['fr', 'en'])
print("✅ Processeur OCR prêt!")


def allowed_file(filename):
    """Vérifie si le fichier a une extension autorisée"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Traite l'upload et exécute l'OCR"""
    
    # Vérifier qu'un fichier a été envoyé
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier envoyé'}), 400
    
    file = request.files['file']
    
    # Vérifier que le fichier a un nom
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400
    
    # Vérifier l'extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Type de fichier non autorisé'}), 400
    
    try:
        # Sauvegarder le fichier
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = Path(app.config['UPLOAD_FOLDER']) / unique_filename
        file.save(str(filepath))
        
        print(f"📁 Fichier sauvegardé: {filepath}")
        
        # Exécuter l'OCR
        results = ocr_processor.process_image(
            filepath, 
            output_dir=app.config['OUTPUT_FOLDER']
        )
        
        # Créer la visualisation
        ocr_processor.visualize_results(
            filepath, 
            results, 
            output_dir=app.config['OUTPUT_FOLDER']
        )
        
        # Préparer la réponse
        response = {
            'success': True,
            'filename': unique_filename,
            'results': results['ocr_results'],
            'annotated_image': f"{filepath.stem}_annotated.jpg"
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/output/<path:filename>')
def download_file(filename):
    """Permet de télécharger les fichiers de sortie"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Sert les images uploadées"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/health')
def health():
    """Endpoint de santé pour vérifier que l'API fonctionne"""
    return jsonify({
        'status': 'ok',
        'ocr_engines': ['easyocr', 'tesseract'],
        'languages': ocr_processor.languages
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Serveur OCR démarré!")
    print("="*60)
    print("📍 URL: http://localhost:5000")
    print("💡 Appuyez sur Ctrl+C pour arrêter")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
