# Shakti 🌿

**Shakti** is a local-first business helper for women running small vegetable-selling businesses. It helps track daily stock, record customer orders, learn customer favourites from order history, and prepare personalised WhatsApp messages in Marathi, Hindi, or English.

## Features

- Daily stock entry and inventory view
- Customer list stored in a local SQLite database
- Paste a customer's order and parse common English/Marathi vegetable names
- Reduce stock when an order is recorded
- Personalised message templates based on available stock and order history
- WhatsApp click-to-send links: the seller reviews the text and taps to open WhatsApp
- Optional Gemma integration for order extraction and message writing
- Regex/template fallback so the core app can work without downloading a model

## Run locally

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000.

## Optional Gemma

Gemma is opt-in so the app can run without a large model download.

1. Accept the model terms on Hugging Face for `google/gemma-3-1b-it`.
2. Install the optional dependencies:

```bash
pip install -r requirements-gemma.txt
```

3. Set environment variables before starting the server.

Windows PowerShell:
```powershell
$env:SHAKTI_USE_GEMMA="1"
$env:HF_TOKEN="hf_your_token"
$env:GEMMA_MODEL="google/gemma-3-1b-it"
uvicorn app:app --reload
```

macOS/Linux:
```bash
export SHAKTI_USE_GEMMA=1
export HF_TOKEN=hf_your_token
export GEMMA_MODEL=google/gemma-3-1b-it
uvicorn app:app --reload
```

The first Gemma call downloads model files and can be slow. After starting the app, open `/api/gemma/status`, then use the interactive `/docs` page to run `POST /api/gemma/test`. A successful response contains `"ok": true`. The fallback can still run if Gemma fails, so check this test endpoint rather than assuming AI is active just because the dashboard works. For a smaller machine, leave Gemma disabled and use the fallback. A larger model may require a capable GPU and more memory.

If `/api/orders` says Method Not Allowed in a browser, it is normally a sign that an older/different server process is running: this project defines `GET /api/orders`. Stop Uvicorn with Ctrl+C, open a terminal in the folder that contains this `app.py`, and run `python -m uvicorn app:app --reload`. Check `/api/health` and `/docs` to confirm the running app and available methods. Use `POST /api/orders/preview` to preview an order, then `POST /api/orders` to save seller-confirmed items; simply opening the URL in a browser sends GET.

## Data and privacy

By default, the database is `shakti.db` beside `app.py`. Customer names, phone numbers, orders, and stock are stored locally. The app does not call a third-party API in fallback mode. If Gemma is enabled, model files are downloaded from Hugging Face on first use and inference runs in the Python process; no hosted inference API is configured by this project.

WhatsApp links open WhatsApp with a prefilled message. The seller still reviews and sends it herself. Avoid entering customer data on a shared device, and keep backups of `shakti.db`.

## Current prototype limits

- The fallback order parser recognises a small set of vegetable names and common quantity formats; always check the parsed order before packing.
- The parser currently reduces stock only for stock items whose names match the parsed item.
- This is a prototype, not an accounting, payment, or food-safety system.
- Message templates are starter copy and should be reviewed for natural language and local dialect before use with customers.

## Project structure

```text
shakti/
├── app.py
├── requirements.txt
├── README.md
└── static/
    └── index.html
```

## Why open innovation?

The core workflow can run on an ordinary laptop without a paid AI API. Local SQLite storage keeps customer records on the seller's device, and an open-weight model can be inspected, swapped, or adapted for local language needs. A fallback keeps the basic workflow available even when Gemma is not installed.
