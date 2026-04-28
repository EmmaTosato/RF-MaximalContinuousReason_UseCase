# Run dataset e dump Redis

Questo file descrive `run_dataset_and_dump.py`, uno script unico per:

1. caricare un dataset baseline;
2. inizializzare Redis con training set, Random Forest, endpoints e candidati iniziali;
3. lanciare i worker;
4. attendere la fine dell'elaborazione;
5. salvare il dump completo dei database Redis selezionati.

Lo script riusa la logica gia presente in `init_baseline.py`, `launch_workers.py`,
`experiments_utils.py` e `redis_backup.py`.

## Requisiti

- Redis raggiungibile su `localhost:6379`, oppure host/porta passati da CLI.
- Dipendenze Python installate da `requirements.txt`.
- Dataset e classificatore presenti sotto:
  - `baseline/resources/datasets/<dataset>/<dataset>.csv`
  - `baseline/resources/datasets/<dataset>/<dataset>.samples`
  - `baseline/Classifiers-100-converted/<dataset>/*.json`

Per vedere i dataset disponibili:

```bash
python run_dataset_and_dump.py --list-datasets
```

## Uso rapido

Esegue `iris` sulla classe `0`, con 4 worker, e salva il dump dei DB Redis `0..10`:

```bash
python run_dataset_and_dump.py iris --class-label 0 --workers 4
```

Esegue tutte le classi predette del dataset:

```bash
python run_dataset_and_dump.py wine-recog --workers 8 --sample-timeout 600
```

Imposta un timeout globale e ferma eventuali worker gia registrati:

```bash
python run_dataset_and_dump.py sonar --class-label 1 --workers 8 --run-timeout 3600 --stop-existing-workers
```

## Flusso interno

Per ogni classe richiesta lo script:

1. connette Redis e, di default, svuota i DB mappati dal progetto;
2. carica dataset, samples e classificatore baseline;
3. converte la Random Forest nel formato interno;
4. scrive in Redis `TRAINING_SET`, `RF`, `EU`, `label`, metadata e candidati iniziali in `CAN`/`PR`;
5. lancia `worker_cache_logged.py` tramite `WorkerManager`;
6. aspetta che i worker terminino, o li ferma se `--run-timeout` scade;
7. crea un dump binario-ripristinabile con `redis_backup`;
8. crea anche `redis_dump_readable.json`, salvo `--no-readable-dump`.

Se non passi `--class-label`, lo script scopre le classi predette dal classificatore
e le esegue una alla volta. Ogni classe riparte da una inizializzazione Redis pulita,
a meno che venga passato `--no-clean`.

## Output

La directory predefinita e:

```text
results/dataset_runs/<dataset>/<run_id>/
```

Per ogni classe viene creata una sottodirectory:

```text
class_<label>/
  manifest.json
  redis_backup_db0.json
  redis_backup_db1.json
  ...
  redis_backup_db10.json
  redis_dump_readable.json
  run_metadata.json
  logs/
```

Nel root della run viene scritto:

```text
dataset_run_metadata.json
```

I file `redis_backup_db*.json` sono dump basati sul comando Redis `DUMP` e sono
pensati per il restore con le funzioni di `redis_backup.py`. Il file
`redis_dump_readable.json` serve invece per ispezione manuale.

## Opzioni principali

- `--class-label`: classe da eseguire; ripetibile o separata da virgola.
- `--workers`: numero di worker.
- `--sample-timeout`: timeout per singolo sample, passato ai worker.
- `--run-timeout`: timeout globale della classe; allo scadere ferma i worker e fa comunque il dump.
- `--databases`: DB Redis da esportare, default `0-10`.
- `--output-dir`: directory base per i risultati, default `results/dataset_runs`.
- `--run-id`: nome esplicito della directory di run.
- `--no-readable-dump`: non genera il dump leggibile.
- `--no-clean`: non svuota Redis prima dell'inizializzazione.
- `--stop-existing-workers`: ferma i worker gia presenti in `workers/worker_pids.json`.
- `--no-use-R-cache`, `--no-use-GP-cache`, `--no-use-NR-cache`, `--no-use-BP-cache`: disabilitano i rispettivi check/cache nei worker.

## Restore del dump

Esempio Python:

```python
from pathlib import Path
from redis_backup import load_multi_database_backup_from_directory, restore_multi_database_backup

backup_dir = Path("results/dataset_runs/iris/20260428_120000/class_0")
backups = load_multi_database_backup_from_directory(backup_dir)
restore_multi_database_backup(backups, {"host": "localhost", "port": 6379}, flush_each=True)
```

`flush_each=True` svuota ogni DB prima del restore.

## Visualizzazione stile `reasons_analysis.ipynb`

Il file `view_reason_analysis.py` permette di generare un report HTML con le
visualizzazioni usate nel notebook `reasons_analysis.ipynb`, partendo da una
directory di checkpoint/dump Redis.

Lo script:

1. carica i file `redis_backup_db*.json` con `etl.loader.etl_from_dir`;
2. estrae test samples, training set, feature names ed endpoints universe;
3. calcola le sigma con `cost_function.cal_sigmas`;
4. calcola i costi per `reasons`, `non_reasons` e `anti_reasons`;
5. calcola la robustness per sample e per anti-reason;
6. genera un report HTML con grafici Plotly;
7. salva anche i CSV intermedi.

Esempio su una run generata da `run_dataset_and_dump.py`:

```bash
python view_reason_analysis.py --checkpoint-dir results/dataset_runs/iris/20260428_120000/class_0
```

Se vuoi far cercare automaticamente i checkpoint sotto `results`:

```bash
python view_reason_analysis.py --results-dir results --checkpoint-index 0
```

Per forzare un dataset name nel report:

```bash
python view_reason_analysis.py --checkpoint-dir results/dataset_runs/iris/20260428_120000/class_0 --dataset-name iris
```

Per visualizzare un sample specifico:

```bash
python view_reason_analysis.py --checkpoint-dir results/dataset_runs/iris/20260428_120000/class_0 --sample-id iris_0_0
```

Per limitare il numero di bitmap elaborate per tipo, utile su dump grandi:

```bash
python view_reason_analysis.py --checkpoint-dir results/dataset_runs/iris/20260428_120000/class_0 --max-bitmaps-per-type 100
```

Output predefinito:

```text
results/reason_visualizations/<dataset>/<checkpoint>/
  reason_analysis_report.html
  reason_costs.csv
  sample_robustness.csv
  anti_reasons_robustness.csv
```

Il report HTML include:

- distribuzione della sample-level robustness;
- distribuzione per quartili;
- visualizzazione del sample con maximal reason;
- visualizzazione del sample con anti-reason;
- confronto reason vs anti-reason;
- confronto smooth corridor;
- anti-reason corridor colorato.

Nota: le figure sono generate con Plotly, quindi `requirements.txt` include
`plotly>=5.0.0`.
