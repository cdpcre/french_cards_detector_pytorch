# Guida al Training in Cloud (Gratis)

Per velocizzare il training rispetto al Mac (MPS), l'opzione migliore è utilizzare GPU NVIDIA (CUDA) disponibili gratuitamente su piattaforme come **Google Colab** o **Kaggle**.

Ecco le due migliori opzioni gratuite:

## Opzione 1: Kaggle Kernels (Consigliata)
Kaggle offre GPU P100 (molto potenti) e 30 ore di utilizzo settimanale gratuito. È spesso più stabile di Colab Free.

### Passaggi:
1.  **Prepara i Dati**:
    *   Carica la cartella `datasets/unified` come nuovo "Dataset" su Kaggle.
    *   Vai su [Kaggle Datasets](https://www.kaggle.com/datasets) -> "New Dataset".
    *   Trascina la cartella o lo zip dei tuoi dati unificati.
2.  **Crea il Notebook**:
    *   Crea un nuovo Notebook su Kaggle.
    *   Nella barra laterale destra, sotto "Accelerator", seleziona **GPU P100**.
    *   Sotto "Input", aggiungi il dataset che hai appena creato.
3.  **Setup Ambiente**:
    ```python
    !pip install ultralytics
    ```
4.  **Esegui il Training**:
    *   Copia il contenuto di `train_custom.py` in una cella (o caricalo come script).
    *   Modifica i percorsi o passa gli argomenti corretti. I dati su Kaggle si trovano solitamente in `/kaggle/input/nome-tuo-dataset/...`.
    *   Esegui il comando:
    ```python
    !python train_custom.py --data /kaggle/input/tuo-dataset/data.yaml --device cuda --epochs 50
    ```

## Opzione 2: Google Colab (Free Tier)
Facile da usare se hai già i dati su Google Drive.

### Passaggi:
1.  **Prepara i Dati**:
    *   Comprimi la cartella `datasets/unified` in un file `.zip`.
    *   Carica lo zip su Google Drive.
2.  **Setup Notebook**:
    *   Apri un nuovo notebook Colab.
    *   Vai su "Runtime" -> "Change runtime type" -> Seleziona **T4 GPU**.
3.  **Monta Drive e Estrai**:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    
    !unzip -q /content/drive/MyDrive/path/to/unified.zip -d /content/data
    !pip install ultralytics
    ```
4.  **Esegui il Training**:
    *   Carica il tuo file `train_custom.py` nella sezione file di Colab (a sinistra).
    *   Esegui:
    ```python
    !python train_custom.py --data /content/data/unified/data.yaml --device cuda --epochs 50
    ```

## Modifiche Consigliate allo Script
Per assicurarti che `train_custom.py` funzioni bene su Linux/Cloud, assicurati di passare sempre `--device cuda` quando lo lanci, altrimenti potrebbe cercare di usare `mps` (Mac) se è il default hardcodato, o `cpu`.

Il tuo script attuale ha questa logica:
```python
parser.add_argument('--device', type=str, default='mps', help='Device (cuda/mps/cpu)')
```
Ricordati di specificare `--device 0` o `--device cuda` nel comando di avvio su Cloud.
