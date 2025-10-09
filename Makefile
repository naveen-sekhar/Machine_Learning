# Makefile for ML Pipeline
.PHONY: install train test clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies"
	@echo "  train      - Train the machine learning model"
	@echo "  test       - Run tests (placeholder)"
	@echo "  clean      - Clean generated files"
	@echo "  all        - Run complete pipeline (install + train)"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r app/requirements.txt

# Create necessary directories
setup:
	mkdir -p model Results

# Train the model
train: setup
	cd app && python train_fixed.py

# Run tests (placeholder - you can add actual tests later)
test:
	@echo "Running tests..."
	@echo "No tests defined yet. Add your test commands here."

# Clean generated files
clean:
	rm -rf model/*.joblib
	rm -rf Results/*.txt
	rm -rf Results/*.png
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Run complete pipeline
all: install train
	@echo "Pipeline completed successfully!"

# Deploy to Hugging Face Hub
hf-login:
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF_TOKEN) --add-to-git-credential

push-hub:
	huggingface-cli upload naveen-sekhar/drug-classification ./README.md --repo-type=space --commit-message="Update README"
	huggingface-cli upload naveen-sekhar/drug-classification ./app --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload naveen-sekhar/drug-classification ./model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload naveen-sekhar/drug-classification ./Results --repo-type=space --commit-message="Sync Results"

deploy: hf-login push-hub
	@echo "Deployment to Hugging Face completed!"