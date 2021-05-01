
# TO DO

## GAN Eğitme

- **StyleGAN2:** Barış çalıştırıyor iCartoonFace için
- **ALAE:** Çağhan çalıştıracak iCartoonFace için

## Siamese

### **Data Okuma:** 

Gürkan hazırlayacak, DataLoader yapısı kullanılabilir
- iCartoonFace: https://github.com/luxiangju-PersonAI/iCartoonFace

### **Network Oluşturma:** 

Gürkan bakacak, yararlı bazı çalışmalar:

- [*Unsupervised Face Recognition*](https://arxiv.org/pdf/1803.01260.pdf)

- [Graph Based Unsupervised Feature Aggregation](https://openaccess.thecvf.com/content_ICCVW_2019/papers/LSR/Cheng_A_Graph_Based_Unsupervised_Feature_Aggregation_for_Face_Recognition_ICCVW_2019_paper.pdf)

- [One Shot Learning for Face Recognition](https://github.com/avillemin/One-Shot-Learning-for-Face-Recognition)

- [Danbooru 2020 Zero-shot Anime Character Identification Dataset](https://github.com/kosuke1701/ZACI-20-dataset)

- [Measure Face Similarity Based on Deep Learning](http://www.diva-portal.se/smash/get/diva2:1361888/FULLTEXT01.pdf)

## LSTM Yapısı Oluşturma

Sonra yapılacak.

### NEW TASKS 1-9.05.2021

- [] dataset teki sequential panelleri bulalım - hard data processing 
- [] Data Okuma => DataLoader yapıcaz => X: [B, P(panel sayısı), 3, W, H], Y: Masklenmiş yüz, [B, 1 (single face), 3, W, H], (Yüzleri keserken square kesicez yüzü içeren ve 64 * 64 e resize edicez), y resize edilmiş image olarak verilecek. 
	- [] Yüz keserken minimum edge 32 olsun, daha küçük olunca yok olmuş olacak
	- [] DataLoader yapılıcak.
	- [] Average Aspect Ratio bulucaz dataset içinde daha sonra image kesmeyi buna göre yapabiliriz. ve bunu center crop yaparak daha sonra image larda kullanıcaz. 
- [] CNN LSTM networku kurma
- [] Latent alındığı zaman generator discriminator yapısını oluşturma
