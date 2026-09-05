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
  semplice errore pixel-per-pixel, con l'aspettativa (nota in letteratura,
  Bergmann et al. 2018) di localizzazioni più pulite rispetto a un
  autoencoder con sola loss MSE. L'ablation (vedi sotto) conferma questa
  aspettativa solo in parte: aiuta il pixel-level nella maggioranza delle
  categorie testate, ma su `grid` e `tile` la stessa componente SSIM
  destabilizza l'ottimizzazione fino a un collasso del decoder — vedi
  "Limiti noti".

Alternative considerate e scartate (ragionamento a priori, prima di avere
dati — l'ablation `mse_only` discussa in "Ablation" e "Limiti noti" mostra
poi il quadro reale: confermato nella maggioranza delle categorie testate,
con eccezioni concrete):
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
  stopping (patience 20), seed globale fisso per riproducibilità. Sono
  valori standard di letteratura, non ricavati da un tuning sistematico
  per questo progetto — scelta deliberata: con 15 modelli indipendenti da
  allenare, Adam più lo scheduler e l'early stopping si autocorreggono
  (riducono il learning rate, si fermano quando serve) rendendo il valore
  esatto di partenza meno critico di quanto lo sarebbe con SGD, dove un
  tuning fine servirebbe davvero.
- L'early stopping richiede un miglioramento di almeno
  `early_stopping_min_delta` (default `1e-4`) per resettare il contatore
  della pazienza: senza questa soglia, categorie con una loss quasi piatta
  (es. `wood`) trovano "miglioramenti" di ordine `1e-5` che azzerano il
  contatore all'infinito, impedendo all'early stopping di attivarsi anche
  quando la loss si è di fatto già stabilizzata.
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

Due varianti allenate sulle stesse categorie di ablation, per isolare
l'effetto di ogni scelta progettuale:

1. **Loss SSIM+MSE vs solo MSE**: quanto (e in che direzione) la componente
   SSIM cambia la localizzazione del difetto rispetto al solo errore
   pixel-per-pixel — l'ipotesi di design è che aiuti, ma va verificato, non
   assunto (risultato reale: effetto misto, vedi "Limiti noti").
2. **Con denoising vs senza rumore in training**: quanto (e se) l'aggiunta
   di rumore in training migliora la capacità del modello di NON
   ricostruire fedelmente i difetti mai visti.

Le categorie di ablation sono 7, in due gruppi distinti:

- **5 originali** (`carpet`, `leather`, `bottle`, `screw`, `transistor`),
  scelte alla cieca come mix rappresentativo di texture e oggetti rigidi,
  prima di vedere qualunque risultato.
- **2 aggiunte in seguito** (`grid`, `tile`), non alla cieca ma perché
  mostrano un fallimento specifico della loss `ssim_mse` (collasso del
  decoder, vedi "Limiti noti"). Tenute concettualmente separate dalle 5
  originali nel report: includerle nel confronto rappresentativo dopo aver
  visto che avvantaggiano `mse_only` sarebbe una selezione post-hoc del
  campione, non una conferma della tendenza generale.

Il notebook di report (`notebook/report.ipynb`) confronta i risultati del
modello principale, di entrambe le ablation, e mostra qualche esempio
visivo e le curve di training.

## Limiti noti

### Limite strutturale: lo score "massimo" non regge con un errore di ricostruzione sistematico

Tre categorie hanno ROC-AUC image-level vicino o sotto il livello del caso:
`pill` (0.423), `screw` (0.455), `metal_nut` (0.490) — nonostante un
ROC-AUC pixel-level buono o eccellente per tutte e tre (0.913, 0.967,
0.862). Non è un caso isolato di `screw`: è lo stesso meccanismo che si
ripete con cause superficiali diverse.

Verificato quantitativamente confrontando lo score (massimo della mappa di
anomalia) tra immagini "good" e difettose dello stesso test set:

| Categoria | Score medio "good" | Score medio difettose |
|---|---|---|
| `pill` | 1.084 | 1.059 |
| `metal_nut` | 1.635 | 1.624 |

Le due distribuzioni sono praticamente sovrapposte — le immagini "good"
hanno score medio addirittura leggermente più alto delle difettose. Lo
score massimo non ha quasi nessuna relazione con la presenza reale di un
difetto.

Causa comune: il bottleneck da 100 numeri non riesce a ricostruire
fedelmente un dettaglio fine ma sistematico della categoria, producendo un
errore di ricostruzione presente su **ogni** immagine, sana o difettosa,
che domina lo score massimo e ne annulla il potere discriminante. Il
dettaglio che sfugge cambia da categoria a categoria:

- **`pill`**: macchioline/speckle di colore naturali sulla superficie,
  visibili anche nelle immagini "good", che il modello appiattisce — la
  mappa di anomalia si accende su decine di questi punti indipendentemente
  dal vero difetto.
- **`metal_nut`**: bordi e riflessi metallici difficili da ricostruire con
  precisione — la mappa si accende lungo tutto il contorno, sia su pezzi
  sani che difettosi.
- **`screw`**: a differenza di quanto assunto sopra, `screw` non ha una
  posa realmente fissa tra le immagini (la vite compare ruotata/traslata),
  e il bottleneck non riesce a ricostruire fedelmente la filettatura fine,
  producendo un errore di ricostruzione diffuso lungo tutto il bordo.

`leather` (blur della texture, vedi "Approccio") e `carpet` (image AUROC
0.577, la seconda peggiore dopo questo terzetto) condividono probabilmente
la stessa famiglia di causa, senza averla verificata con la stessa
profondità.

Su `screw` sono stati testati tre approcci per calibrare lo score
(baseline media sul train, filtro passa-alto gaussiano, erosione
morfologica): i primi due hanno aiutato alcune categorie rigide (`bottle`,
`transistor`) ma nessuno ha risolto `screw` in modo soddisfacente, e il
filtro passa-alto ha addirittura peggiorato le altre categorie. Si è
quindi mantenuto lo score originale (massimo della mappa raw). Una
soluzione reale richiederebbe catturare il dettaglio fine specifico di
ogni categoria (bottleneck più ampio, loss percettiva, o
allineamento/registrazione dell'oggetto per `screw`) — fuori scope per
questo progetto.

### `grid` e `tile`: collasso del decoder sotto `ssim_mse`

Allenando `grid` e `tile` con la loss `ssim_mse` (sia in configurazione
`main` sia in `no_denoising`, che usa comunque `ssim_mse`), il **decoder
collassa**: dopo 1-3 epoche smette di dipendere dal vettore latente e
produce sempre la stessa ricostruzione, quasi piatta, indipendentemente
dall'immagine in input. Verificato empiricamente confrontando le
ricostruzioni di immagini diverse: l'encoder continua a produrre latent
diversi da immagine a immagine, il decoder no. La loss si blocca su un
valore alto (~0.66-0.68) e non scende più.

Non è un minimo naturale della loss: lo score SSIM tra la ricostruzione
collassata e l'immagine reale è basso (0.17-0.24), molto peggio di una
categoria sana come `bottle` (0.94-0.96) — è un blocco dell'ottimizzazione
(il gradiente verso il decoder si azzera), non una scorciatoia premiata
dalla loss. Causa più probabile: saturazione del `BatchNorm` nel decoder,
più facile da innescare su categorie a bassa varianza locale (texture
molto regolari).

Passando a `mse_only` il collasso sparisce su entrambe le categorie, ma
l'effetto sull'AUROC è sorprendentemente diverso: su `grid` migliora
nettamente (image AUROC 0.793 → 0.840), su `tile` invece **peggiora**
(0.909 → 0.610) nonostante il decoder torni a funzionare correttamente.
Spiegazione più plausibile: un decoder collassato produce comunque
un'immagine di riferimento fissa, e lo score di anomalia diventa di fatto
"quanto l'immagine di test si discosta da questo riferimento fisso" — un
meccanismo degenere ma non privo di segnale, che su `tile` (difetti ampi e
contrastati) separa comunque bene normale da anomalo quasi per caso, su
`grid` (difetti più sottili) no. Lezione: un image-AUROC alto da un modello
con decoder collassato non prova che il modello funzioni come inteso — va
sempre verificato che la ricostruzione dipenda davvero dall'input.

Non risolto nel codice principale (richiederebbe cambiare la loss del
modello per tutte le 15 categorie, perdendo la coerenza con `bottle` e
`leather` dove `ssim_mse` funziona bene) — documentato come limite noto,
con `grid`/`tile` allenate specificamente in `mse_only` per l'ablation.

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
