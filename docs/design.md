# Design del progetto: Anomaly Detection su MVTec AD

## Obiettivo

Un autoencoder allenato per categoria del dataset MVTec AD, capace di
rilevare difetti su prodotti industriali sia a livello di intera immagine
(normale/anomalo) sia a livello di pixel (localizzazione del difetto),
sfruttando le maschere `ground_truth` già presenti nel dataset. Il progetto
include anche un notebook di report con analisi comparativa dei risultati.

## Dataset

Il file `mvtec_anomaly_detection.tar.xz` contiene il dataset MVTec AD
standard: 15 categorie (bottle, cable, capsule, carpet, grid, hazelnut,
leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood,
zipper). Ogni categoria ha:

- `train/good/*.png` — solo immagini senza difetti, usate per l'addestramento
  (paradigma unsupervised).
- `test/<good|tipo_difetto>/*.png` — immagini di test, sia normali che con
  vari tipi di difetto.
- `ground_truth/<tipo_difetto>/<id>_mask.png` — maschere di segmentazione
  binarie del difetto, una per ogni immagine di test difettosa.

## Approccio: Denoising Convolutional Autoencoder con loss SSIM + MSE

- Autoencoder convoluzionale con bottleneck stretto, senza skip connection.
- Durante il training viene aggiunto rumore gaussiano alle immagini di
  input; il target di ricostruzione resta l'immagine pulita. Questo evita
  che il modello impari una funzione quasi-identità che ricostruirebbe
  troppo bene anche i difetti (rischio concreto su categorie a texture
  ripetitiva come carpet, leather, tile), forzandolo a modellare i pattern
  strutturali della classe "normale".
- La mappa di anomalia è `1 - SSIM_map` tra immagine originale e
  ricostruzione: cattura differenze strutturali/di contrasto invece del
  semplice errore pixel-per-pixel, producendo localizzazioni più pulite
  rispetto a un autoencoder con sola loss MSE (approccio noto in
  letteratura, Bergmann et al. 2018).

Alternative considerate e scartate:
- **Solo MSE**: localizzazione pixel-level più rumorosa sui bordi.
- **Skip connection (stile U-Net)**: rischio che il difetto passi
  attraverso le skip connection e venga ricostruito troppo fedelmente,
  vanificando la detection.
- **VAE**: guadagno marginale per questo caso d'uso, a fronte di un tuning
  più delicato (bilanciamento del termine KL).

## Architettura del modello

Input normalizzato in [0,1], risoluzione fissa 256×256×3 per tutte le
categorie.

Encoder (conv stride 2, BatchNorm, LeakyReLU):

```
3x256x256 -> Conv(32, k4 s2 p1)   -> 32x128x128
          -> Conv(64, k4 s2 p1)   -> 64x64x64
          -> Conv(128, k4 s2 p1)  -> 128x32x32
          -> Conv(256, k4 s2 p1)  -> 256x16x16
          -> Conv(512, k4 s2 p1)  -> 512x8x8
          -> Conv(latent_dim=100, k8 s1 p0) -> 100x1x1   (bottleneck)
```

Decoder: speculare con `ConvTranspose2d`, BatchNorm, ReLU, fino a tornare a
`3x256x256`, attivazione finale `Sigmoid`. `latent_dim` è configurabile
(default 100).

## Loss e rumore

- Rumore: gaussiano additivo (deviazione standard di default 0.1),
  applicato solo al batch di training, mai a validazione/test.
- Loss di training: `loss = 0.85 * (1 - SSIM) + 0.15 * MSE`, pesi
  configurabili. La componente MSE stabilizza il training nelle prime
  epoche, dove la SSIM da sola può avere gradiente poco informativo su
  regioni uniformi. La SSIM è implementata direttamente nel progetto
  (finestra gaussiana 11×11) invece di dipendere da una libreria esterna,
  perché serve sia come loss di training sia come mappa di anomalia in
  valutazione — deve essere esattamente la stessa funzione in entrambi i
  casi.
- I due comportamenti (rumore, tipo di loss) sono controllati da flag di
  configurazione, così le due ablation riusano lo stesso codice di
  training senza duplicazione.

## Pipeline dati

- Split: `train/good` diviso 90/10 in train/validation (seed fisso),
  usato solo per monitorare la loss e fare early stopping — nessuna
  immagine difettosa entra mai nel training.
- Test set: tutte le sottocartelle di `test/`, ciascuna immagine associata
  all'etichetta binaria (good=0, qualsiasi tipo di difetto=1) e, se
  disponibile, alla maschera `ground_truth` corrispondente.
- Preprocessing: resize a 256×256, normalizzazione in [0,1]. Nessuna data
  augmentation geometrica: molte categorie MVTec hanno oggetti con posa
  fissa tra train e test (es. bottle, cable). Non tutte: `screw` in
  particolare ha rotazione/posizione variabile da un'immagine all'altra —
  vedi la sezione "Limiti noti" più sotto per l'impatto che questo ha sulla
  valutazione.

## Training

- Un modello indipendente per categoria (stessa architettura, stessi
  iperparametri di default), pesi non condivisi tra categorie.
- Iperparametri di default: batch size 32, ottimizzatore Adam (lr=1e-3),
  scheduler `ReduceLROnPlateau` (dimezza il learning rate se la loss di
  validazione ristagna per 10 epoche), massimo 200 epoche con early
  stopping (patience 20), seed globale fisso per riproducibilità.
- Checkpoint salvato in `outputs/<categoria>/model.pt` (solo il modello
  con la migliore loss di validazione); storia di training in
  `outputs/<categoria>/history.csv`.

## Valutazione e metriche

- Mappa di anomalia per immagine di test: `1 - SSIM_map` tra immagine e
  sua ricostruzione.
- Score immagine-level: massimo della mappa di anomalia → **ROC-AUC
  image-level** per categoria (distingue immagini normali da difettose).
- Score pixel-level: la mappa di anomalia confrontata pixel per pixel con
  la maschera reale → **ROC-AUC pixel-level** per categoria (misura la
  qualità della localizzazione).
- Soglia binaria: 95° percentile della distribuzione degli score
  immagine-level sul train set "good" — permette una decisione concreta
  normale/anomalo oltre all'AUROC continuo.
- Output visivo: per un campione di immagini di test, griglia con
  originale, ricostruzione, mappa di anomalia e maschera reale.

## Ablation (analisi comparativa)

Due varianti allenate sulle stesse 5 categorie rappresentative (carpet,
leather, bottle, screw, transistor — un mix di texture e oggetti rigidi),
per isolare l'effetto di ogni scelta progettuale:

1. **Loss SSIM+MSE vs solo MSE**: quanto la componente SSIM migliora la
   localizzazione del difetto rispetto al solo errore pixel-per-pixel.
2. **Con denoising vs senza rumore in training**: quanto l'aggiunta di
   rumore in training migliora la capacità del modello di NON ricostruire
   fedelmente i difetti mai visti.

Il notebook di report (`notebook/report.ipynb`) confronta i risultati del
modello principale, di entrambe le ablation, e mostra qualche esempio
visivo e le curve di training.

## Limiti noti

`screw` (e in misura minore `carpet`) hanno ROC-AUC image-level vicino o
sotto il livello del caso (0.42 e 0.58), nonostante un ROC-AUC pixel-level
eccellente per `screw` (0.97): la mappa di anomalia localizza correttamente
il difetto quando c'è, ma lo score a livello di immagine intera (il massimo
della mappa) non separa bene le immagini normali da quelle difettose.

Causa: a differenza di quanto assunto sopra, `screw` non ha una posa
realmente fissa tra le immagini (la vite compare ruotata/traslata), e il
bottleneck da 100 numeri non riesce a ricostruire fedelmente la filettatura
fine, producendo un errore di ricostruzione diffuso lungo tutto il bordo —
alto sia su viti sane sia difettose. Il massimo della mappa non distingue
questo errore diffuso "normale" da un vero difetto puntiforme.

Sono stati testati tre approcci per calibrare lo score (baseline media sul
train, filtro passa-alto gaussiano, erosione morfologica): i primi due
hanno aiutato alcune categorie rigide (`bottle`, `transistor`) ma nessuno
ha risolto `screw` in modo soddisfacente, e il filtro passa-alto ha
addirittura peggiorato le altre categorie. Si è quindi mantenuto lo score
originale (massimo della mappa raw). Una soluzione reale richiederebbe
l'allineamento/registrazione dell'oggetto prima del confronto — fuori scope
per questo progetto.

## Struttura del progetto

```
PROGETTO/
├── data/                        # dataset estratto (non versionato)
├── mvtec_anomaly_detection.tar.xz   # archivio originale (non versionato)
├── config/default.yaml          # iperparametri di default (rispecchia Config)
├── data_classes/mvtec_dataset.py    # MVTecTrainDataset, MVTecTestDataset
├── model_classes/autoencoder_model.py   # ConvAutoencoder
├── anomaly_ae/                  # config, loss, ablation, training/valutazione, report
├── scripts/extract_dataset.py   # estrazione dell'archivio del dataset
├── train.py                     # entry point: allena UNA categoria (--category/--ablation)
├── test.py                      # entry point: valuta UNA categoria (--category/--ablation)
├── run_all.sh                   # loop su tutte le 15 categorie + le 2 ablation
├── prepare.sh                   # setup ambiente (installa dipendenze, estrae il dataset)
├── outputs/                     # checkpoint, metriche, immagini di esempio (non versionato)
├── notebook/report.ipynb        # report finale: tabelle, grafici, confronto ablation
├── tests/                       # test automatici (pytest)
├── requirements.txt
└── README.md
```
