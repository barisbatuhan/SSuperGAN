export PYTHONPATH := ${PYTHONPATH}:$(shell pwd)

train_vae:
	python3 playground/vae/vae_playground.py

train_introvae:
	python3 playground/intro_vae/intro_vae_playground.py

train_ssupervae:
	python3 playground/ssupervae/ssupervae_playground.py

train_vaegan:
	python3 playground/vae_gan/train.py

train_seq_vaegan:
	python3 playground/seq_vae_gan/train.py

train_ssuper_dcgan:
	python3 playground/ssuper_dcgan/ssuper_dcgan_play.py

train_ssuper_global_dcgan:
	python3 playground/ssuper_global_dcgan/ssuper_global_dcgan_play.py

train_face_cloze:
	python3 playground/face_cloze/face_cloze_play.py
    
eval_ssupervae:
	python3 playground/ssupervae/eval_ssupervae.py

eval_ssuper_dcgan:
	python3 playground/ssuper_dcgan/eval_ssuper_dcgan.py

eval_ssuper_global_dcgan:
	python3 playground/ssuper_global_dcgan/eval_ssuper_global_dcgan.py
