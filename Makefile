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
	@if not exist "model" mkdir model
	@if not exist "Results" mkdir Results

# Train the model
train: setup
	cd app && python train_fixed.py

# Run tests (placeholder - you can add actual tests later)
test:
	@echo "Running tests..."
	@echo "No tests defined yet. Add your test commands here."

# Clean generated files
clean:
	@if exist "model\*.joblib" del /q "model\*.joblib"
	@if exist "Results\*.txt" del /q "Results\*.txt"
	@if exist "Results\*.png" del /q "Results\*.png"
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@del /s /q *.pyc 2>nul || echo No .pyc files found

# Run complete pipeline
all: install train
	@echo "Pipeline completed successfully!"

# Deploy to Hugging Face Hub
hf-login:
	git pull origin update || true
	git switch update || true
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF_TOKEN) --add-to-git-credential

push-hub:
	huggingface-cli upload naveen-sekhar/drug-classification ./app --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload naveen-sekhar/drug-classification ./model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload naveen-sekhar/drug-classification ./Results --repo-type=space --commit-message="Sync Results"

deploy: hf-login push-hub
	@echo "Deployment to Hugging Face completed!"