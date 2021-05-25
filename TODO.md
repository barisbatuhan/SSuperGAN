
# TO DO

### ROADMAP and MODELS

- [+] AutoEncoder Like Structure 
- VAE Like Structure ( with/out LSTM or Simple Concating )
- VAE Like Structure With a GAN
- VAE like Structure With a BiGAN or GAN and a separete enconding of face 
- Inpainting Baseline
- Possible Combo of All of these


### NEW TASKS 20-27.05.2021

- **[Barış]** DCGAN yapısı düzgün çalışacak hale getirilmeli, genel yüz yapısı, face orientation, color info falan tutturulmalı
- **[Gürkan]** Inpainting içindeki Fine Pass çıkartılarak Global ve Local Discr. Coarse generator'a bağlanmalı, reconstruction kalitesi
- **[Çağhan]** StyleGAN implementasyonu bizim yapıya benzetip golden yüz datasıyla eğitmek e.g. [Style-Based-GAN](https://github.com/rosinality/style-based-gan-pytorch).
- **[Barış & Çağhan]** DCGAN Loss yapısını WGAN Loss tarzına çevirmek
- **[Gürkan]** Grid-/RandomSearch tarzında bir yapı ile doğru hyper-parameter arama.


## CNN Backbone Yapısı Eğitimi

- **[Barış]** Binary person classifier eğitme yapılabilir EfficientNet kullanarak.
- **[Gürkan]** DINO / BYOL gibi 
- **[Gürkan]** Inpainting yapısı ile pretrain yapmak.

## Golden Dataset Üzerine Geliştirmeler

- **[+]** Bir preprocess ile şu yapı çıkartılacak: 

```python
{
    1: [
        ["img1.jpg", "img2.jpg", "..."],
        [
            ["<img1_panel_coords>", "<img1_face_coords>"],
            ["<img2_panel_coords>", "<img2_face_coords>"],
            "..."
        ]
        
    ],
    
    2: "..."
}
```

- **[+]** Daha sade ve sadece crop & augment olan bir dataset çıkartılacak.

## Model Geliştirmeleri

- **[Çağhan]** GAN yapısı oluşturup (DCGAN gibi ve daha da advanced 1 tane - [In-Domain Inversion](https://github.com/genforce/idinvert_pytorch)) yüz datasıyla eğitme.
- Global discriminator yapısı ekleme ve onunla modeli eğitme.
- **[Gürkan]** Weight save ettikten sonra o saved weight ile training sürecine devam ettirmek. [+]
- **[İleri Seviye]** LSTM oluşturma.


### NEW TASKS 1-9.05.2021

- [+] Dataset teki sequential panelleri bulalım - hard data processing 
- [+] Data Okuma => DataLoader yapıcaz => X: [B, P(panel sayısı), 3, W, H], Y: Masklenmiş yüz, [B, 1 (single face), 3, W, H], (Yüzleri keserken square kesicez yüzü içeren ve 64 * 64 e resize edicez), y resize edilmiş image olarak verilecek. 
	- [+] Yüz keserken minimum edge 32 olsun, daha küçük olunca yok olmuş olacak
	- [+] DataLoader yapılıcak.
	- [+] Average Aspect Ratio bulucaz dataset içinde daha sonra image kesmeyi buna göre yapabiliriz. ve bunu center crop yaparak daha sonra image larda kullanıcaz. 
- [+] CNN LSTM networku kurma
- [+] Latent alındığı zaman generator discriminator yapısını oluşturma




### **Network Oluşturma:**  [+]

Gürkan bakacak, yararlı bazı çalışmalar:

- [*Unsupervised Face Recognition*](https://arxiv.org/pdf/1803.01260.pdf)

- [Graph Based Unsupervised Feature Aggregation](https://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Cheng_A_Graph_Based_Unsupervised_Feature_Aggregation_for_Face_Recognition_ICCVW_2019_paper.pdf)

- [One Shot Learning for Face Recognition](https://github.com/avillemin/One-Shot-Learning-for-Face-Recognition)

- [Danbooru 2020 Zero-shot Anime Character Identification Dataset](https://github.com/kosuke1701/ZACI-20-dataset)

- [Measure Face Similarity Based on Deep Learning](http://www.diva-portal.se/smash/get/diva2:1361888/FULLTEXT01.pdf)


## Siamese

### **Data Okuma:** [+]

Gürkan hazırlayacak, DataLoader yapısı kullanılabilir 
- iCartoonFace: https://github.com/luxiangju-PersonAI/iCartoonFace
