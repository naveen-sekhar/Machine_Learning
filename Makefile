install:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python3 train.py

eval:
	echo "## Model Metrics" > report.md
	cat Results/metrics.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name "naveensekhar"
	git config --global user.email "naveensekhar1547@gmail.com"
	git add Model Results
	git commit -m "Update model and results"
	git push --force origin HEAD:update

hf-login:
	git pull origin update || true
	git switch update || true
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload yourHFusername/DRUG_CLASSIFY ./App --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload yourHFusername/DRUG_CLASSIFY ./Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload yourHFusername/DRUG_CLASSIFY ./Results --repo-type=space --commit-message="Sync Results"

deploy: hf-login push-hub
