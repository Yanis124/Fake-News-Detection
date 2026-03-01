# Fake News Detection

Un projet de machine learning pour détecter les fausses nouvelles en utilisant le corpus MultiNLI et diverses techniques de traitement du langage naturel.

## 📋 Table des matières

- [À propos](#à-propos)
- [Ensemble de données](#ensemble-de-données)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Modèles et résultats](#modèles-et-résultats)
- [Notebooks](#notebooks)
- [Licence](#licence)

## À propos

Ce projet vise à développer et évaluer différents modèles de machine learning pour détecter les fausses nouvelles. Le projet utilise l'inférence en langage naturel (NLI) pour classifier les relations entre les prémisses et les hypothèses, ce qui aide à identifier la désinformation.

## Ensemble de données

Le projet utilise le **MultiNLI (Multi-Genre Natural Language Inference) Corpus 1.0**.


**Données fournies :**
- `multinli_1.0_train.jsonl` - Ensemble d'entraînement
- `multinli_1.0_dev_matched.jsonl` - Ensemble de validation (matched)
- `multinli_1.0_dev_mismatched.jsonl` - Ensemble de test (mismatched)


## Prérequis

- Python 3.8+
- pip ou conda

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Yanis124/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Créer un environnement virtuel

```bash
# Avec venv
python -m venv .venv
.venv\Scripts\activate  # Sur Windows
source .venv/bin/activate  # Sur Linux/Mac

# Ou avec conda
conda create -n fake-news python=3.8
conda activate fake-news
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## Structure du projet

```
Fake-News-Detection/
├── README.md                      # Cette fichier
├── requirements.txt               # Dépendances Python
├── data/
│   ├── README.txt                # Info sur le corpus MultiNLI
│   ├── raw/                      # Fichiers bruts du corpus
│   └── processed/                # Données traitées
├── src/                          # Code source principal
├── notebook/                     # Notebooks Jupyter
│   ├── analysis.ipynb            # Analyse exploratoire
│   ├── Feature_engineering_importance.ipynb
│   ├── Fine_Tunning.ipynb
│   ├── TF-IDF_engineering.ipynb
│   └── TF-IDF_ML.ipynb
├── exp_results/                  # Résultats des expériences
│   ├── artifacts/                # Modèles et vectorizers
│   ├── features_importance/      # Importance des features
│   ├── fine_tuning/              # Résultats du fine-tuning
│   └── tf-idf_enginering/        # Résultats TF-IDF
└── .venv/                        # Environnement virtuel
```

## Utilisation

### Exécuter les notebooks

```bash
jupyter notebook notebook/
```

**Notebooks disponibles :**

1. **analysis.ipynb** - Analyse exploratoire des données
2. **TF-IDF_engineering.ipynb** - Ingénierie des features avec TF-IDF
3. **TF-IDF_ML.ipynb** - Entraînement et évaluation des modèles
4. **Feature_engineering_importance.ipynb** - Analyse de l'importance des features
5. **Fine_Tunning.ipynb** - Fine-tuning des modèles

### Exécuter depuis le terminal

```bash
# Activer l'environnement virtuel
.venv\Scripts\activate

# Lancer Jupyter
jupyter notebook

# Ou utiliser des scripts Python (s'ils existent dans src/)
python src/your_script.py
```

## Modèles et résultats

Les résultats des expériences sont stockés dans `exp_results/`:

- **artifacts/** - Modèles entraînés, vectorizers
- **metrics/** - Métriques de performance (accuracy, precision, recall, F1-score)
- **features_importance/** - Importance des features pour chaque expérience
- **tf-idf_enginering/** - Résultats des expériences TF-IDF
- **fine_tuning/** - Résultats du fine-tuning des hyperparamètres

### Accédez aux métriques

Les fichiers de métriques sont au format JSON et incluent des informations sur:
- Accuracy
- Precision
- Recall
- F1-score
- Autres métriques de classification

## Notebooks

### analysis.ipynb
Analyse exploratoire des données du corpus MultiNLI incluant:
- Distribution des labels
- Longueur des phrases
- Statistiques descriptives

### TF-IDF_engineering.ipynb
Préparation des données avec TF-IDF:
- Nettoyage du texte
- Vectorization avec TF-IDF
- Configuration du preprocessor

### TF-IDF_ML.ipynb
Entraînement et évaluation des modèles:
- Pipeline d'entraînement
- Évaluation multimodèle
- Comparaison des performances

### Feature_engineering_importance.ipynb
Analyse de l'importance des features:
- Features les plus pertinentes
- Visualisations
- Insights

### Fine_Tunning.ipynb
Optimisation des hyperparamètres:
- Sélection du meilleur hyperparamètres

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

Si vous utilisez le corpus MultiNLI, veuillez citer:

```bibtex
@InProceedings{williams2018broad,
  author    = {Williams, Adina and Nangia, Nikita and Bowman, Samuel R.},
  title     = {A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference},
  booktitle = {Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year      = {2018},
  publisher = {Association for Computational Linguistics},
}
```

---
