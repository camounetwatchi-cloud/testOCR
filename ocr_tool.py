#!/usr/bin/env python3
"""
Outil OCR pour détecter du texte manuscrit et imprimé
Supporte Tesseract et EasyOCR
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import easyocr
import pytesseract
from PIL import Image
import numpy as np


class OCRProcessor:
    """Classe principale pour le traitement OCR"""
    
    def __init__(self, languages=['fr', 'en']):
        """
        Initialise les moteurs OCR
        
        Args:
            languages: Liste des langues à détecter (ex: ['fr', 'en'])
        """
        self.languages = languages
        print(f"🔄 Initialisation d'EasyOCR avec les langues: {languages}")
        self.easyocr_reader = easyocr.Reader(languages, gpu=False)
        print("✅ EasyOCR initialisé")
    
    def preprocess_image(self, image_path):
        """
        Prétraite l'image pour améliorer l'OCR
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Image prétraitée (numpy array)
        """
        # Lire l'image
        img = cv2.imread(str(image_path))
        
        if img is None:
            raise ValueError(f"Impossible de lire l'image: {image_path}")
        
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Amélioration du contraste avec CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Débruitage
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Binarisation adaptative
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        return binary
    
    def ocr_with_tesseract(self, image_path, preprocess=True):
        """
        Exécute Tesseract OCR
        
        Args:
            image_path: Chemin vers l'image
            preprocess: Appliquer le prétraitement
            
        Returns:
            Dict avec le texte et les détails
        """
        print("🔄 Exécution de Tesseract OCR...")
        
        if preprocess:
            img = self.preprocess_image(image_path)
        else:
            img = Image.open(image_path)
        
        # Configuration Tesseract pour manuscrit
        custom_config = r'--oem 3 --psm 6 -l fra'
        
        # Extraction du texte
        text = pytesseract.image_to_string(img, config=custom_config)
        
        # Extraction des données détaillées
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=custom_config)
        
        # Filtrer les détections avec confiance > 30
        results = []
        for i, conf in enumerate(data['conf']):
            if int(conf) > 30:
                results.append({
                    'text': data['text'][i],
                    'confidence': int(conf),
                    'bbox': {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
                })
        
        print(f"✅ Tesseract: {len(results)} éléments détectés")
        
        return {
            'engine': 'tesseract',
            'full_text': text.strip(),
            'detections': results
        }
    
    def ocr_with_easyocr(self, image_path, preprocess=True):
        """
        Exécute EasyOCR (meilleur pour le manuscrit)
        
        Args:
            image_path: Chemin vers l'image
            preprocess: Appliquer le prétraitement
            
        Returns:
            Dict avec le texte et les détails
        """
        print("🔄 Exécution d'EasyOCR...")
        
        if preprocess:
            img = self.preprocess_image(image_path)
        else:
            img = cv2.imread(str(image_path))
        
        # Détection du texte
        results_raw = self.easyocr_reader.readtext(img, paragraph=False)
        
        # Formater les résultats
        results = []
        full_text_parts = []
        
        for detection in results_raw:
            bbox, text, confidence = detection
            
            # Calculer la bounding box
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            results.append({
                'text': text,
                'confidence': float(confidence) * 100,
                'bbox': {
                    'x': int(min(x_coords)),
                    'y': int(min(y_coords)),
                    'width': int(max(x_coords) - min(x_coords)),
                    'height': int(max(y_coords) - min(y_coords))
                }
            })
            
            full_text_parts.append(text)
        
        print(f"✅ EasyOCR: {len(results)} éléments détectés")
        
        return {
            'engine': 'easyocr',
            'full_text': ' '.join(full_text_parts),
            'detections': results
        }
    
    def process_image(self, image_path, output_dir='output', use_both=True):
        """
        Traite une image avec les deux moteurs OCR
        
        Args:
            image_path: Chemin vers l'image
            output_dir: Dossier de sortie
            use_both: Utiliser les deux moteurs
            
        Returns:
            Dict avec tous les résultats
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n📸 Traitement de: {image_path.name}")
        print("=" * 60)
        
        results = {
            'image': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'ocr_results': {}
        }
        
        # EasyOCR (meilleur pour le manuscrit)
        try:
            easyocr_results = self.ocr_with_easyocr(image_path)
            results['ocr_results']['easyocr'] = easyocr_results
        except Exception as e:
            print(f"❌ Erreur EasyOCR: {e}")
            results['ocr_results']['easyocr'] = {'error': str(e)}
        
        # Tesseract
        if use_both:
            try:
                tesseract_results = self.ocr_with_tesseract(image_path)
                results['ocr_results']['tesseract'] = tesseract_results
            except Exception as e:
                print(f"❌ Erreur Tesseract: {e}")
                results['ocr_results']['tesseract'] = {'error': str(e)}
        
        # Sauvegarder les résultats
        output_file = output_dir / f"{image_path.stem}_ocr_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Sauvegarder le texte brut
        text_file = output_dir / f"{image_path.stem}_text.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("EASYOCR (recommandé pour manuscrit)\n")
            f.write("=" * 60 + "\n\n")
            if 'easyocr' in results['ocr_results'] and 'full_text' in results['ocr_results']['easyocr']:
                f.write(results['ocr_results']['easyocr']['full_text'])
            
            if use_both and 'tesseract' in results['ocr_results']:
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("TESSERACT\n")
                f.write("=" * 60 + "\n\n")
                if 'full_text' in results['ocr_results']['tesseract']:
                    f.write(results['ocr_results']['tesseract']['full_text'])
        
        print(f"\n💾 Résultats sauvegardés:")
        print(f"   - JSON: {output_file}")
        print(f"   - Texte: {text_file}")
        
        return results
    
    def visualize_results(self, image_path, results, output_dir='output'):
        """
        Crée une visualisation avec les bounding boxes
        
        Args:
            image_path: Chemin vers l'image
            results: Résultats OCR
            output_dir: Dossier de sortie
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        
        # Lire l'image
        img = cv2.imread(str(image_path))
        
        if img is None:
            return
        
        # Dessiner les bounding boxes EasyOCR
        if 'easyocr' in results['ocr_results'] and 'detections' in results['ocr_results']['easyocr']:
            for detection in results['ocr_results']['easyocr']['detections']:
                bbox = detection['bbox']
                x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                
                # Rectangle vert pour EasyOCR
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Texte avec confiance
                label = f"{detection['confidence']:.0f}%"
                cv2.putText(img, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Sauvegarder l'image annotée
        output_file = output_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(output_file), img)
        print(f"   - Image annotée: {output_file}")


def main():
    """Fonction principale en ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Outil OCR pour texte manuscrit et imprimé'
    )
    parser.add_argument('image', help='Chemin vers l\'image à analyser')
    parser.add_argument('--output', '-o', default='output', 
                       help='Dossier de sortie (défaut: output)')
    parser.add_argument('--lang', '-l', nargs='+', default=['fr', 'en'],
                       help='Langues à détecter (défaut: fr en)')
    parser.add_argument('--tesseract-only', action='store_true',
                       help='Utiliser uniquement Tesseract')
    parser.add_argument('--no-viz', action='store_true',
                       help='Ne pas créer la visualisation')
    
    args = parser.parse_args()
    
    # Vérifier que l'image existe
    if not Path(args.image).exists():
        print(f"❌ Erreur: L'image '{args.image}' n'existe pas")
        sys.exit(1)
    
    # Créer le processeur OCR
    processor = OCRProcessor(languages=args.lang)
    
    # Traiter l'image
    results = processor.process_image(
        args.image, 
        output_dir=args.output,
        use_both=not args.tesseract_only
    )
    
    # Créer la visualisation
    if not args.no_viz:
        processor.visualize_results(args.image, results, output_dir=args.output)
    
    print("\n✅ Traitement terminé avec succès!")
    
    # Afficher un aperçu du texte détecté
    if 'easyocr' in results['ocr_results'] and 'full_text' in results['ocr_results']['easyocr']:
        print("\n📝 Aperçu du texte détecté (EasyOCR):")
        print("-" * 60)
        text = results['ocr_results']['easyocr']['full_text']
        print(text[:500] + ('...' if len(text) > 500 else ''))


if __name__ == '__main__':
    main()
