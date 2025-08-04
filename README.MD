# MCOC Arena Bot - Action Predictor

Un sistema di intelligenza artificiale per predire le azioni nel gioco Marvel Contest of Champions (MCOC) utilizzando reti neurali convoluzionali (CNN) e ricorrenti (RNN).

## 🧠 Architettura del Modello

Il sistema utilizza un'architettura **CNN + RNN** per predire 5 azioni possibili:

- `light_attack` (a)
- `special` (s) 
- `block` (d)
- `evade` (w)
- `medium_attack` (space)

### Struttura del Modello

```
[Frame_1] [Frame_2] ... [Frame_10]
    ↓         ↓              ↓
[ CNN Backbone (MobileNetV2) ]
    ↓         ↓              ↓
[Feature_1] [Feature_2] ... [Feature_10]
                  ↓
            [GRU (2 layer)]
                  ↓
          [Dense + Softmax]
                  ↓
           [Predicted Action]
```

**Caratteristiche:**
- **Input**: Sequenze di 10 frame (2.5 secondi a 4 FPS)
- **CNN**: MobileNetV2 preaddestrato per estrazione feature
- **RNN**: GRU a 2 layer per apprendimento temporale
- **Output**: Predizione ogni 200-250ms su 5 classi

## 📁 Struttura del Progetto

```
mcoc-arena-bot/
├── assets/
│   ├── config/           # Configurazioni del gioco
│   └── data/
│       └── raw/          # Screenshot numerati con label
├── models/               # Modelli addestrati
├── results/              # Risultati e grafici
├── scripts/
│   ├── train_model.py    # Training del modello
│   ├── inference.py      # Inferenza in tempo reale
│   └── test_model.py     # Test e analisi
└── requirements.txt      # Dipendenze Python
```

## 🚀 Installazione

1. **Clona il repository:**
```bash
git clone <repository-url>
cd mcoc-arena-bot
```

2. **Installa le dipendenze:**
```bash
pip install -r requirements.txt
```

3. **Prepara i dati:**
   - Metti i tuoi screenshot in `assets/data/raw/`
   - I file devono essere numerati sequenzialmente (es. `003631_a.png`)
   - La label è nel nome del file dopo l'underscore

## 🎯 Utilizzo

### 1. Training del Modello

```bash
cd scripts
python train_model.py
```

**Parametri configurabili in `train_model.py`:**
- `SEQUENCE_LENGTH`: Lunghezza sequenza (default: 10)
- `BATCH_SIZE`: Dimensione batch (default: 16)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `NUM_EPOCHS`: Numero epoche (default: 30)

**Output:**
- Modello salvato in `models/best_model.pth`
- Grafici di training in `results/`
- Report delle performance

### 2. Test del Modello

```bash
python test_model.py
```

Analizza:
- Distribuzione delle classi nei dati
- Performance del modello
- Predizioni temporali

### 3. Inferenza in Tempo Reale

#### Da Webcam:
```bash
python inference.py --mode camera
```

#### Da Video:
```bash
python inference.py --mode video --source video.mp4
```

#### Da Screenshot:
```bash
python inference.py --mode screenshots --screenshot_dir assets/data/raw
```

**Controlli:**
- `q`: Esci
- `s`: Salva frame corrente

## 📊 Risultati

Il sistema genera automaticamente:

1. **Grafici di Training:**
   - Loss e accuracy nel tempo
   - Confusion matrix

2. **Analisi Dati:**
   - Distribuzione delle classi
   - Performance per classe

3. **Predizioni Temporali:**
   - Sequenze di azioni nel tempo
   - Confidence scores

## 🔧 Configurazione Avanzata

### Modifica Architettura

In `train_model.py`, puoi modificare:

```python
class Config:
    SEQUENCE_LENGTH = 10      # Lunghezza sequenza
    FEATURE_DIM = 256         # Dimensioni feature CNN
    HIDDEN_SIZE = 128         # Dimensioni hidden GRU
    DROPOUT_RATE = 0.3        # Dropout per regolarizzazione
```

### Cambiare Backbone CNN

```python
# In MCOCActionPredictor.__init__()
self.cnn = models.efficientnet_b0(pretrained=True)  # Invece di MobileNetV2
```

### Usare TCN invece di GRU

```python
# Sostituisci GRU con TCN per convoluzioni temporali
self.tcn = TemporalConvNet(...)
```

## 🎮 Integrazione con il Bot

Il modello addestrato può essere integrato nel bot esistente:

```python
from scripts.inference import MCOCPredictor

predictor = MCOCPredictor("models/best_model.pth")
action, confidence = predictor.predict_action(frame)
```

## 📈 Performance Tipiche

Con dati ben bilanciati:
- **Accuracy**: 85-95%
- **FPS**: 4-5 (inferenza)
- **Latency**: 200-250ms per predizione

## 🛠️ Troubleshooting

### Problemi Comuni:

1. **CUDA out of memory:**
   - Riduci `BATCH_SIZE`
   - Usa `feature_dim` più piccolo

2. **Overfitting:**
   - Aumenta `DROPOUT_RATE`
   - Aggiungi data augmentation

3. **Basse performance:**
   - Verifica bilanciamento classi
   - Aumenta `SEQUENCE_LENGTH`
   - Usa più dati di training

### Debug:

```bash
# Test rapido su pochi dati
python test_model.py

# Verifica distribuzione classi
python -c "from scripts.test_model import analyze_data_distribution; analyze_data_distribution()"
```

## 🤝 Contribuire

1. Fork il progetto
2. Crea un branch per la feature
3. Commit le modifiche
4. Push al branch
5. Crea una Pull Request

## 📄 Licenza

Questo progetto è sotto licenza MIT. Vedi `LICENSE` per dettagli.

## 🙏 Ringraziamenti

- PyTorch per il framework di deep learning
- OpenCV per la computer vision
- La community MCOC per i dati di training