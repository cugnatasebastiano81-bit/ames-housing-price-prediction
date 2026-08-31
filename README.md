# Previsione dei prezzi immobiliari con Ames Housing

Prototipo accademico di regressione tabellare per stimare `SalePrice` sul dataset Ames Housing.
Il progetto separa sviluppo e holdout, apprende il preprocessing all'interno della pipeline,
confronta più modelli con cross-validation e riusa lo stesso oggetto addestrato per l'inferenza.

## Contenuto

- notebook eseguito con analisi, selezione del modello e valutazione finale;
- modulo Python importabile con validazione, preprocessing, training e inferenza;
- controlli automatici sui contratti dei dati e sui casi limite;
- versioni delle dipendenze fissate per Python 3.12;
- verifica SHA-256 del dataset prima del caricamento.

## Requisiti

- Python `3.12.x`; l'ambiente di riferimento usa Python `3.12.13`;
- ambiente CPU;
- accesso personale a Kaggle per ottenere il dataset.

Creazione di un ambiente virtuale su Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Su Linux o macOS, il comando di attivazione equivalente è:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ottenere il dataset

Il dataset non è incluso nel repository perché la pagina Kaggle sorgente mostra la licenza come
`Unknown`.

1. Aprire la pagina [Ames Housing Dataset su Kaggle](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset).
2. Accedere con il proprio account e scaricare `AmesHousing.csv` secondo le condizioni applicabili.
3. Copiare il file nella cartella principale del progetto.
4. Verificare che lo SHA-256 sia:

```text
65A1CBB89C2B58B11674097135DD04E710DF5DB726BCFE6EC551DE38B44E4911
```

In PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath .\AmesHousing.csv
```

Il file è escluso da Git tramite `.gitignore`. Per provenienza, limiti di redistribuzione e
citazione consultare l'avviso sul dataset.

## Uso

Aprire `AmesHousing_Pipeline.ipynb` in un ambiente Jupyter compatibile ed eseguire le celle in ordine,
dall'inizio alla fine, senza stato precedente.

La funzione pubblica di inferenza è:

```python
from ames_pipeline import predict_price

prezzo = predict_price(dati_immobile)
```

`dati_immobile` deve contenere esattamente le 79 feature grezze previste dal contratto. `SalePrice`,
`Order` e `PID` non sono accettati come input di inferenza.

## Controlli locali

Dopo aver collocato il dataset nella cartella principale:

```powershell
python -m pytest
python -m ruff check .
```

## Riproducibilità e limiti

Lo split, i seed, gli inventari degli iperparametri e le dipendenze dirette sono fissati. Il notebook
documenta hash del dataset, digest dello split e metriche osservate.

Il limite metodologico principale è storico: lo stesso holdout era già stato osservato in una versione
precedente del lavoro. Il protocollo corrente evita leakage procedurale e non usa il test per scegliere
il modello, ma le metriche finali non sostituiscono una futura verifica su dati etichettati realmente
nuovi.

Il progetto è un esercizio accademico e non costituisce uno strumento professionale di valutazione
immobiliare, consulenza finanziaria o stima certificata.

## Licenza

Il software e la documentazione originale del progetto sono distribuiti con licenza MIT. Il dataset
non è incluso e non è coperto dalla licenza MIT del progetto.
