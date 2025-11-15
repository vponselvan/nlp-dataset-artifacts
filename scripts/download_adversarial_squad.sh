#!/bin/bash

# Download SQuAD Adversarial dataset (AddSent)
# From the original paper: https://github.com/robinjia/adversarial-squad

ADDSENT_URL="https://raw.githubusercontent.com/robinjia/adversarial-squad/master/data/dev-v1.1.json"
OUTPUT_DIR="./data"
OUTPUT_FILE="$OUTPUT_DIR/squad_adversarial.json"

echo "Downloading SQuAD Adversarial (AddSent) dataset..."

mkdir -p "$OUTPUT_DIR"

# Try wget first, then curl
if command -v wget &> /dev/null; then
    wget -O "$OUTPUT_FILE" "$ADDSENT_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$OUTPUT_FILE" "$ADDSENT_URL"
else
    echo "❌ Error: Neither wget nor curl is available"
    echo "Please install one of them or download manually from:"
    echo "  $ADDSENT_URL"
    exit 1
fi

if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✅ Downloaded successfully to: $OUTPUT_FILE"
    echo ""
    echo "Dataset statistics:"
    python3 -c "
import json
with open('$OUTPUT_FILE') as f:
    data = json.load(f)
    num_articles = len(data['data'])
    num_paragraphs = sum(len(article['paragraphs']) for article in data['data'])
    num_qas = sum(len(qa['qas']) for article in data['data'] for qa in article['paragraphs'])
    print(f'  Articles: {num_articles}')
    print(f'  Paragraphs: {num_paragraphs}')
    print(f'  Questions: {num_qas}')
"
    echo ""
    echo "Use this file with: --dataset $OUTPUT_FILE"
else
    echo "❌ Download failed"
    exit 1
fi
