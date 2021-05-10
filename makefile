export PYTHONPATH := ${PYTHONPATH}:$(shell pwd)

train_vae:
	python3 playground/vae/vae_playground.py

train_ssupervae:
	python3 playground/ssupervae/plain_ssupervae_playground.py