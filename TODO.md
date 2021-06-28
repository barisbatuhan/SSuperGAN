
# TO DO

## Future Works

- DCGAN Loss yapısını WGAN Loss tarzına çevirmek
- DINO / BYOL gibi 
- Inpainting yapısı ile pretrain yapmak.
- Binary person classifier eğitme yapılabilir EfficientNet kullanarak.

## Genel

- **[Herkes]** [AE Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html) oku, tartış.

## 28.06.2021

- **[Barış & Çağhan]** StyleGAN implementasyonu bizim yapıya benzetip golden yüz datasıyla eğitmek e.g. 
	- [Style-Based-GAN](https://github.com/rosinality/style-based-gan-pytorch)
	- [In-Domain Inversion](https://github.com/genforce/idinvert_pytorch)
- **[Gürkan]** Siyah-beyaz olarak resmi process et. Bunun için resim başta Gray sonra RGB çevrilip process edilebilir ya da EfficientNet için input olarak num_channels var ise o değiştirilebilir. Ayrıca GAN yapısının output channel size da dynamic olmalı ya da ilk channel alınmalı sadece.
- **[Gürkan]** 1 resim ve 3 resim ile eğitimin görsel sonuç karşılaştırması yapılmalı.
- **[Herkes]** Character Identification ile ilgili labeling yapılabilir ya da MTurk için bir arayüz hazırlanabilir.

## 21.06.2021

- **[Gürkan]** Grid-/RandomSearch tarzında bir yapı ile doğru hyper-parameter arama.
- **[Gürkan]** Model yapısını PyTorch Lightning ile çevirme (başlangış SSuperVAE).
- **[+]** VAE-GAN yapısı iyice eğitilip KL loss ile düzgün çalışır hale getirilmeye çalışılacak. Sonrasında Sequential Encoder yapısı ile bu modül bağlanacak.
- **[+]** Jigsaw model epitimini tamamla ve yardımcı olabilir mi bak. --> Köşelerebakıp objelere odaklanmıyor.
- **[+]** Düz panel resimleri döndüren bir Dataset yapısı oluşturulacak, sequence length bir parametre olacak ona göre return edilecek.

## 20.05.2021

- **[+]** DCGAN yapısı düzgün çalışacak hale getirilmeli, genel yüz yapısı, face orientation, color info falan tutturulmalı
- **[+]** Inpainting içindeki Fine Pass çıkartılarak Global ve Local Discr. Coarse generator'a bağlanmalı, reconstruction kalitesi

## Golden Dataset Üzerine Geliştirmeler

- **[+]** Bir preprocess ile sequential panel yapısı ve panel/face annotation yapısını geliştir. 
- **[+]** Daha sade ve sadece crop & augment olan bir dataset çıkartılacak.

## Model Geliştirmeleri

- **[+]** GAN yapısı oluşturup (DCGAN) yüz datasıyla eğitme.
- **[+]** Global discriminator yapısı ekleme ve onunla modeli eğitme.
- **[+]** Weight save ettikten sonra o saved weight ile training sürecine devam ettirmek. [+]
- **[+]** LSTM oluşturma.

## 01.05.2021

- **[+]** Dataset teki sequential panelleri bulalım - hard data processing 
- **[+]** Data Okuma => DataLoader yapıcaz => X: [B, P(panel sayısı), 3, W, H], Y: Masklenmiş yüz, [B, 1 (single face), 3, W, H], (Yüzleri keserken square kesicez yüzü içeren ve 64 * 64 e resize edicez), y resize edilmiş image olarak verilecek. 
	- **[+]** Yüz keserken minimum edge 32 olsun, daha küçük olunca yok olmuş olacak
	- **[+]** DataLoader yapılıcak.
	- **[+]** Average Aspect Ratio bulucaz dataset içinde daha sonra image kesmeyi buna göre yapabiliriz. ve bunu center crop yaparak daha sonra image larda kullanıcaz. 
- **[+]** CNN LSTM networku kurma
- **[+]** Latent alındığı zaman generator discriminator yapısını oluşturma

## **[+]** **Network Oluşturma:**

Gürkan bakacak, yararlı bazı çalışmalar:

- [*Unsupervised Face Recognition*](https://arxiv.org/pdf/1803.01260.pdf)

- [Graph Based Unsupervised Feature Aggregation](https://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Cheng_A_Graph_Based_Unsupervised_Feature_Aggregation_for_Face_Recognition_ICCVW_2019_paper.pdf)

- [One Shot Learning for Face Recognition](https://github.com/avillemin/One-Shot-Learning-for-Face-Recognition)

- [Danbooru 2020 Zero-shot Anime Character Identification Dataset](https://github.com/kosuke1701/ZACI-20-dataset)

- [Measure Face Similarity Based on Deep Learning](http://www.diva-portal.se/smash/get/diva2:1361888/FULLTEXT01.pdf)


## **[+]** **Siamese - Data Okuma**

Gürkan hazırlayacak, DataLoader yapısı kullanılabilir 
- iCartoonFace: https://github.com/luxiangju-PersonAI/iCartoonFace
