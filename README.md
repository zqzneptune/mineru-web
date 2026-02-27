# MinerU-Web

A lightweight Flask-based web interface for the [MinerU](https://github.com/opendatalab/MinerU) PDF-to-Markdown extraction engine. It provides a simple GUI to upload PDFs, stream processing logs, and browse extracted results.

## 1. Install MinerU

Follow the MinerU installation guidelines.

## 2. Install Web UI
Clone this repository and install the web requirements:

```bash
git clone https://github.com/your-username/mineru-web.git
cd mineru-web
pip install flask
```

## 3. Run
Start the application:

```bash
python app.py
```
Open your browser to `http://localhost:5000`.

## Memory Note
This version is optimized for **8GB GPUs**. It uses a subprocess `spawn` method and `0.75` GPU utilization to ensure text extraction and formula OCR can both fit in VRAM.
