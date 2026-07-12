# Server upload and resumable execution

Extract `lncrna_server_package.tar.gz`, enter the extracted repository, install `requirements.txt`, and run:

```bash
bash server/server_run_concat.sh
```

Then run, in order:

```bash
bash server/server_run_transfer_fast.sh
bash server/server_run_transfer_rflda.sh
bash server/server_run_binary.sh
bash server/server_finalize_after_reruns.sh
```

Every script resumes cell-by-cell, skips a valid 10-fold performance file, uses lock directories, and appends to `results/logs/server_rerun_ledger.tsv`. All outputs remain under `results/`.
