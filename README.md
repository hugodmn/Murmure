# 🧠 Murmure – Audio Transcription Pipeline

**Murmure** est une pipeline modulaire de transcription audio combinant :

- [Whisper](https://github.com/openai/whisper) d’OpenAI pour la reconnaissance vocale  
- [Silero VAD](https://github.com/snakers4/silero-vad) pour la détection d’activité vocale  
- Diarisation des locuteurs (optionnelle)  
- Accélération via `whisper.cpp` sur Apple Silicon (M1/M2)

---

## 📦 Installation

1. Clonez le dépôt :

```bash
git clone https://github.com/hugodmn/murmure.git  
cd murmure
```

2. Installez la bibliothèque localement :

```bash
pip install .
```

> Vous pouvez maintenant exécuter la CLI avec :

```bash
murmure --help
```

---

## 🚀 Utilisation

```bash
murmure path/to/audio.wav --model medium --vad --diarization
```

### Options :

| Argument        | Description                                              |
|----------------|----------------------------------------------------------|
| `path`          | Chemin vers un fichier ou dossier audio (`.wav`, `.mp3`)|
| `--model`       | Taille du modèle Whisper : `tiny`, `base`, `small`, `medium`, `large` |
| `--vad`         | Active la détection d’activité vocale                   |
| `--diarization` | Active la diarisation des locuteurs                     |
| `--device`      | **Optionnel**. Forcer un device : `cpu`, `cuda`, ou `mps` |

> ℹ️ **Le device est automatiquement détecté** (CUDA > MPS > CPU).  
> Utilisez `--device` uniquement si vous souhaitez forcer un comportement spécifique.

---

## 💻 Support des devices

### ✅ Détection automatique

Par défaut, Murmure choisit le meilleur device disponible dans cet ordre :

1. `cuda` (GPU NVIDIA)
2. `mps` (Apple Silicon)
3. `cpu` (fallback)

### ✅ CPU (par défaut si rien d’autre)

```bash
murmure path.wav
```

### ✅ CUDA GPU

```bash
murmure path.wav --device cuda
```

> Nécessite une carte NVIDIA et des drivers CUDA.

### ✅ Apple Silicon (M1/M2) avec MPS

```bash
murmure path.wav --device mps
```

> Pour utiliser `mps`, Murmure s'appuie sur [whisper.cpp](https://github.com/ggerganov/whisper.cpp)

---

## ⚙️ Installation manuelle de `whisper.cpp` (MPS uniquement)

1. Clonez le dépôt officiel :

```bash
git clone https://github.com/ggerganov/whisper.cpp.git  
cd whisper.cpp
```

2. Compilez le projet :

```bash
cmake -B build  
cmake --build build -j
```

3. Rendez le binaire exécutable :

```bash
chmod +x build/bin/whisper-cli
```

4. C’est tout !

Murmure détectera automatiquement le binaire `whisper-cli` si placé dans le dossier `whisper.cpp/build/bin`.  
Sinon, une erreur claire sera levée si vous utilisez `--device mps`.

---

## 🔧 Exemple complet

```bash
murmure /path/to/audio.wav --model small --vad --diarization
```

```bash
murmure /path/to/audio.wav --model base --device mps --vad
```
