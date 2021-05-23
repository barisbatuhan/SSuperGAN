export PYTHONPATH := ${PYTHONPATH}:$(shell pwd)

train_vae:
	python3 playground/vae/vae_playground.py

train_introvae:
	python3 playground/intro_vae/intro_vae_playground.py

train_ssupervae:
	python3 playground/ssupervae/ssupervae_playground.py

eval_ssupervae:
	python3 playground/ssupervae/eval_ssupervae.py
