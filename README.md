# MVTec AD Anomaly Detection

## Descrizione del task e dell'approccio

**Task**: rilevamento di anomalie (difetti) su immagini di prodotti
industriali del dataset [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
(15 categorie: bottle, cable, capsule, carpet, grid, hazelnut, leather,
metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper), sia a
**livello di intera immagine** (normale/anomalo) sia a **livello di pixel**
(localizzazione del difetto). L'approccio è **unsupervised**: il modello si
allena solo su immagini senza difetti ("good") — in produzione reale i
difetti sono rari e di tipo imprevedibile, quindi non si può contare su un
dataset etichettato che li copra tutti.

**Approccio**: un **autoencoder convoluzionale denoising**, allenato
separatamente per ciascuna categoria, con loss combinata **SSIM + MSE**.
Il modello impara a ricostruire immagini "normali"; su un'immagine con un
difetto mai visto la ricostruzione fallisce proprio in quella zona, e la
differenza tra originale e ricostruzione (`1 - SSIM_map`) diventa la mappa
di anomalia. Durante il training si aggiunge rumore gaussiano all'input
(target di ricostruzione resta l'immagine pulita) per impedire una
scorciatoia a funzione-identità che ricostruirebbe fedelmente anche i
difetti. Il progetto include due ablation (loss SSIM+MSE vs solo MSE;
denoising vs senza rumore) per isolare l'effetto di queste due scelte.

Per il design completo (architettura, motivazioni, limiti noti) vedi
[`docs/design.md`](docs/design.md).

## Configurazione dell'ambiente

Richiede Python 3.11+ (sviluppato e testato con Python 3.14) e, per un
training in tempi ragionevoli, una GPU NVIDIA con driver CUDA.

1. Crea e attiva un virtual environment nella cartella del progetto:

       python -m venv .venv

   (su Windows l'interprete si trova poi in `.venv/Scripts/python`; su
   Linux/macOS in `.venv/bin/python`.)

2. Esegui `prepare.sh` (installa le dipendenze, poi estrae il dataset se
   l'archivio è già presente nella root):

       bash prepare.sh

   Su Windows, se `bash` non è disponibile, esegui i due comandi contenuti
   nello script direttamente:

       .venv/Scripts/python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
       .venv/Scripts/python -m pip install -r requirements.txt

   (la wheel di `torch` di default su PyPI per Windows è solo CPU: va
   installata prima la build CUDA, versione fissata; pip vedrà poi `torch`
   già soddisfatto in `requirements.txt` e lo salterà).

3. Verifica che la GPU sia rilevata:

       .venv/Scripts/python -c "import torch; print(torch.cuda.is_available())"

   Se stampa `False` (o se non hai una GPU), la pipeline funziona comunque
   su CPU, ma il training di tutte le 15 categorie richiede diverse ore
   invece di un paio.

## Dataset: download e accesso

Il dataset MVTec AD **non è incluso nel repository** (troppo grande, ~4.9
GB compresso). Per ottenerlo:

1. Scarica l'archivio `mvtec_anomaly_detection.tar.xz` dal sito ufficiale
   MVTec: <https://www.mvtec.com/company/research/datasets/mvtec-ad> (il
   download richiede la compilazione di un modulo per uso di ricerca).
2. Posiziona il file scaricato nella root del progetto (accanto a
   `README.md`), **prima** di eseguire `prepare.sh` (altrimenti esegui
   l'estrazione a mano, vedi sotto).
3. Se non l'hai già fatto tramite `prepare.sh`, estrai l'archivio con:

       .venv/Scripts/python scripts/extract_dataset.py

   Estrae l'archivio in `data/mvtec_ad/<categoria>/`. L'operazione è
   idempotente: se `data/mvtec_ad/` esiste già e contiene file, lo script
   salta l'estrazione invece di rifare 5 GB di lavoro inutilmente.

## Esecuzione: training e valutazione

Coerentemente con il template del corso, i due passi principali sono due
script alla radice del progetto, `train.py` e `test.py`, che allenano/valutano
**una categoria alla volta**. Gli iperparametri di default sono in
`config/default.yaml` (rispecchia i default di `Config`, così puoi cambiarli
senza toccare il codice).

### Allena (`train.py`) e valuta (`test.py`)

    .venv/Scripts/python train.py --config config/default.yaml --category bottle
    .venv/Scripts/python test.py --config config/default.yaml --category bottle

Ablation (categoria dev'essere una delle 7 di ablation — vedi
`docs/design.md`):

    .venv/Scripts/python train.py --config config/default.yaml --category bottle --ablation mse_only
    .venv/Scripts/python test.py --config config/default.yaml --category bottle --ablation mse_only

Scrive `outputs/<categoria>/metrics.json` (e le varianti delle ablation in
`outputs/ablation_mse_only/`, `outputs/ablation_no_denoising/`), oltre alle
immagini di esempio per l'ispezione visiva.

### Tutte le categorie insieme (`run_all.sh`)

    bash run_all.sh

Allena e valuta in sequenza tutte e 15 le categorie del modello principale,
poi le 7 categorie di ablation per entrambe le varianti (`mse_only`,
`no_denoising`) — è lo script usato per produrre tutti i risultati riportati
in questo README e nel notebook.

### Report

    .venv/Scripts/python -m jupyter notebook notebook/report.ipynb

Legge i file `metrics.json` prodotti sopra — esegui prima i passi di
train/test per il modello principale ed entrambe le ablation.

### Test automatici (pytest)

    .venv/Scripts/python -m pytest

## Risultati principali

Modello principale, media sulle 15 categorie: **AUROC image-level 0.721**,
**AUROC pixel-level 0.813** — ma con varianza alta tra categorie (image
AUROC da 0.42 a 0.96): l'affidabilità del metodo dipende molto dal tipo di
prodotto. Migliori: `wood` (0.958 image), `hazelnut` (0.974 pixel).
Peggiori: `pill` (0.423 image), `tile` (0.397 pixel).

**Ablation loss (SSIM+MSE vs solo MSE)**: risultato misto, non una vittoria
netta di una loss sull'altra — `mse_only` è pari o leggermente meglio a
livello immagine, la componente SSIM aiuta leggermente la localizzazione a
livello pixel.

**Ablation denoising (con vs senza rumore)**: effetto piccolo ma
consistente a favore del denoising sull'image-level (+0.023 di media su 7
categorie), trascurabile sul pixel-level.

**Limiti noti** (dettagliati in `docs/design.md`):
- Tre categorie (`pill`, `screw`, `metal_nut`) hanno AUROC image-level
  vicino o sotto il livello del caso nonostante una localizzazione
  pixel-level buona o eccellente: lo score "massimo della mappa" è
  dominato da un errore di ricostruzione sistematico (texture, bordi
  metallici, o posa non fissa a seconda della categoria), presente sia su
  immagini normali che difettose.
- Su `grid` e `tile`, la loss `ssim_mse` causa un collasso del decoder
  (smette di dipendere dall'input, sia in `main` sia in `no_denoising`).
  Non risolto nel modello principale (richiederebbe cambiare la loss per
  tutte le 15 categorie, perdendo la coerenza con `bottle`/`leather` dove
  `ssim_mse` funziona bene). Nell'ablation, passando a `mse_only` il
  collasso sparisce ma l'effetto sull'AUROC è divergente: `grid` migliora
  (0.793 → 0.840), `tile` peggiora (0.909 → 0.610).
