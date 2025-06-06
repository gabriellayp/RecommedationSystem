# 🎬 Film Recommendation System
COHORT ID : MC008D5X2466

Nama : Gabriella Yoanda Pelawi

Email : mc008d5x2466@student.devacademy.id

## 1. 📌 Project Overview

Dalam era digital saat ini, industri hiburan semakin bergantung pada sistem rekomendasi untuk meningkatkan keterlibatan pengguna. Netflix mengklaim bahwa lebih dari 80% jam tayang berasal dari sistem rekomendasi mereka, menandakan betapa krusialnya sistem ini dalam pengalaman pengguna. Meningkatnya jumlah konten membuat pengguna kesulitan dalam memilih film yang relevan dengan preferensi mereka. 

Oleh karena itu, proyek ini bertujuan untuk membangun sistem rekomendasi film yang mampu memberikan rekomendasi yang relevan bagi pengguna berdasarkan riwayat interaksi dan kesamaan konten. Proyek ini penting karena mampu meningkatkan user engagement dan pengalaman pengguna dalam menjelajahi film. Oleh karena itu, kemampuan untuk membangun model rekomendasi yang akurat dan relevan dapat berdampak signifikan secara bisnis.

> Referensi:  
> - [Netflix and the Power of Personalization](https://www.netflixtechblog.com)  

---

## 2. 💼 Business Understanding

### 🎯 Problem Statements
1. Bagaimana memberikan rekomendasi film kepada pengguna yang menyukai film tertentu?
2. Bagaimana memberikan rekomendasi film berdasarkan histori rating pengguna?

### 🥅 Goals
- Menghasilkan rekomendasi film yang relevan untuk pengguna.
- Mengukur performa rekomendasi secara kuantitatif.

### 🧭 Solution Approaches
Kami mengimplementasikan dua pendekatan sistem rekomendasi:
1. **Content-Based Filtering:** Menggunakan kemiripan genre film (TF-IDF + Cosine Similarity).
2. **Collaborative Filtering:** Menggunakan deep learning model berbasis embedding (dengan TensorFlow/Keras).

---

## 3. 🧠 Data Understanding

### 📦 Dataset
Dataset yang digunakan berasal dari Kaggle dengan judul [Movie Lens Small Latest Dataset](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset/data). Dataset ini terdiri dari dua file utama, yaitu `movies.csv` dan `ratings.csv`.

---

### 📄 movies.csv

- Jumlah baris: **9.742** 
- Jumlah kolom: 3 (`movieId`, `title`, `genres`)
- Deskripsi:
  | Fitur     | Deskripsi                                             |
  |-----------|-------------------------------------------------------|
  | `movieId` | ID unik untuk setiap film                            |
  | `title`   | Judul film, termasuk tahun rilis                     |
  | `genres`  | Genre film, dipisahkan oleh simbol pipe `|`          |

- Kondisi Data:
  - Missing Values: Tidak ditemukan missing values.
  - Duplikasi: Tidak ditemukan duplikasi berdasarkan `movieId`.
  - Format genre: Terdapat 20 jenis genre film pada data. Namun, beberapa entri memiliki genre `'no genres listed'`, yang perlu ditangani saat data preparation.

---

### 📄 ratings.csv

- Jumlah baris: **100.836**
- Jumlah kolom: 4 (`userId`, `movieId`, `rating`, `timestamp`)
- Deskripsi:
  | Fitur       | Deskripsi                                                   |
  |-------------|-------------------------------------------------------------|
  | `userId`    | ID unik pengguna                                            |
  | `movieId`   | ID film yang dirating oleh pengguna                         |
  | `rating`    | Nilai rating dari pengguna terhadap film (skala 0.5–5.0)    |
  | `timestamp` | Waktu saat pengguna memberikan rating (format UNIX epoch)   |

- Kondisi Data:
  - Missing Values: Tidak ditemukan missing values.
  - Duplikasi: Tidak ditemukan duplikasi.
  - Tidak ditemukan adanya outlier pada data rating.

### 📊 Exploratory Data Analysis
**1. Distribusi Genre Film 🎬**

   ![Distribusi Genre](https://github.com/gabriellayp/RecommedationSystem/blob/main/images/Genrefilm.png?raw=true)
   
   Genre yang paling mendominasi dalam dataset adalah Drama, diikuti oleh Comedy dan Thriller. Hal ini menunjukkan bahwa film dengan genre drama adalah yang paling sering diproduksi atau tersedia dalam data yang digunakan. Genre-genre seperti IMAX, Film-Noir, dan Western memiliki jumlah film yang jauh lebih sedikit, menunjukkan bahwa film dengan genre tersebut lebih jarang muncul. Distribusi ini dapat memengaruhi performa model, terutama pada sistem rekomendasi berbasis genre seperti content-based filtering, karena genre dominan akan lebih sering direkomendasikan.

**2. Distribusi Rating Film ⭐**

  ![Distribusi Rating](https://github.com/gabriellayp/RecommedationSystem/blob/main/images/ratingfilm.png?raw=true)
  
   Distribusi rating menunjukkan bahwa tidak terdapat outlier, sebagian besar pengguna memberikan rating yang cukup tinggi terhadap film yang mereka tonton. Rating paling umum berada di angka 4.0, diikuti oleh 3.0 dan 5.0. Sebaliknya, rating rendah (seperti 0.5–1.5) jauh lebih jarang diberikan. Hal ini menunjukkan bahwa pengguna cenderung lebih sering menonton dan menilai film yang mereka sukai, atau film dengan kualitas yang relatif baik. Pola ini penting untuk diperhatikan karena bisa menciptakan bias pada model rekomendasi, terutama pada pendekatan collaborative filtering.

---

## 4. 🛠️ Data Preparation

Data preparation dilakukan secara **terpisah** untuk dua pendekatan utama yang digunakan dalam sistem rekomendasi, yaitu **Content-Based Filtering** dan **Collaborative Filtering**. Masing-masing memiliki kebutuhan struktur dan praproses data yang berbeda. 

Namun, sebelumnya dilakukan terlebih dahulu langkah untuk preparation keseluruhan data yang digunakan untuk kedua model yaitu :

- **Menghapus label tidak valid:**  
   Label `(no genres listed)` pada kolom `genres` dihapus karena tidak memberikan informasi yang berguna untuk analisis konten. Jumlah film setelah dibersihkan adalah 9708 film (sebelumnya dibersihkan jumlahnya 9742).


### 📚 A. Content-Based Filtering

Content-Based Filtering fokus pada fitur konten dari item (dalam hal ini: genre film) untuk merekomendasikan film yang mirip berdasarkan profil film itu sendiri.

#### ✅ Langkah-Langkah
**TF-IDF Vectorization:**
   TF-IDF (Term Frequency - Inverse Document Frequency) adalah metode untuk mengubah teks menjadi representasi numerik dengan memberi bobot pada setiap kata berdasarkan seberapa sering kata tersebut muncul dalam dokumen tertentu (Term Frequency) dan seberapa unik kata tersebut dibandingkan seluruh dokumen (Inverse Document Frequency). Pada analisis ini, penerapannya antara lain :
   - Kolom `genres` diubah menjadi representasi numerik menggunakan *TF-IDF Vectorizer*.
   - Tokenisasi dilakukan berdasarkan separator `|` menggunakan `token_pattern=r'[^|]+'`, sehingga setiap genre dianggap sebagai token terpisah.
   - Hasil akhir berupa matriks TF-IDF berukuran `(9708, 19)` — mewakili 9708 film dan 19 genre unik.

#### ⚙️ Alasan
- TF-IDF memberikan bobot proporsional terhadap genre yang lebih unik, menurunkan pengaruh genre populer yang terlalu sering muncul.
- Matriks TF-IDF digunakan untuk menghitung *cosine similarity* antar film, yang menjadi dasar dalam sistem rekomendasi berbasis konten.

---

### 👥 B. Collaborative Filtering

Collaborative Filtering memanfaatkan pola interaksi historis antara pengguna dan item (film) untuk membangun rekomendasi berdasarkan kesamaan preferensi.

#### ✅ Langkah-Langkah
1. **Merge Data:**  
   Dataset `movies.csv` dan `ratings.csv` digabung berdasarkan `movieId` untuk mengaitkan setiap rating dengan informasi judul dan genre film.  
   Hasil penggabungan berisi kolom `userId`, `movieId`, `rating`, `timestamp`, `title`, dan `genres`.

2. **Pivot Table:**  
   Data diubah menjadi matriks dengan `title` sebagai indeks dan `userId` sebagai kolom. Nilai cell menunjukkan rating dari pengguna untuk film tertentu.

3. **Missing Values:**  
   Karena tidak semua pengguna memberi rating untuk semua film, maka muncul nilai `NaN`. Nilai ini diisi dengan 0 yang merepresentasikan ketidakterlibatan (belum dirating).

4. **Filtering:**  
   Hanya menyertakan:
   - Film yang dirating oleh minimal **5 pengguna**.
   - Pengguna yang memberi rating ke minimal **5 film**.  
   Tujuannya adalah untuk meningkatkan kualitas data dan memperkuat sinyal interaksi yang bermakna.

5. **Encoding:**  
   Kolom `userId` dan `movieId` diubah menjadi indeks integer mulai dari 0 (`user_encoded`, `movie_encoded`) untuk digunakan pada model machine learning berbasis embedding.

6. **Normalisasi Rating:**  
   Rating dengan skala asli 0.5–5 dinormalisasi ke dalam rentang 0–1 agar proses pelatihan model lebih stabil dan cepat konvergen.

7. **Transformasi Format Long & Drop Rating 0:**  
   - Matriks pivot dikembalikan ke format long (`userId`, `movieId`, `rating`).
   - Interaksi dengan rating 0 dihapus karena dianggap sebagai non-interaksi yang tidak memberikan sinyal preferensi.

8. **Pembagian Data:**  
   Dataset dibagi menjadi **training set** dan **validation set** menggunakan teknik `GroupShuffleSplit` berdasarkan `userId` agar tidak ada informasi pengguna yang terbagi ke dua set.

#### ⚙️ Alasan
- Penggabungan dataset memastikan bahwa semua informasi user dan film tersedia dalam satu struktur kerja.
- Pivot table mengubah data ke bentuk matriks interaksi user-item yang umum digunakan dalam sistem rekomendasi.
- Imputasi nilai kosong dengan 0 merupakan pendekatan standar pada dataset dengan explicit feedback.
- Encoding diperlukan untuk mengubah nilai kategorikal menjadi bentuk numerik yang dapat diproses model.
- Filtering menjaga kualitas data dengan hanya melibatkan entitas yang aktif dan signifikan.
- Pembagian data berdasarkan `userId` mencegah kebocoran informasi dan menjaga validitas evaluasi model.

Note : Fitur timestamp tidak digunakan dalam analisis karena tidak relevan terhadap tujuan pemodelan (content-based recommendation dan colaborative filtering).

---

## 🤖 Modeling and Results

### 1️⃣ Content-Based Filtering (Genre-Based)

Content-Based Filtering adalah pendekatan rekomendasi yang berfokus pada kemiripan konten antar item. Dalam proyek ini, konten yang dimaksud adalah **genre film**, dan kemiripan antar film dihitung berdasarkan representasi numerik dari genre tersebut.

### 📌 Langkah-Langkah:

1. **Menghitung Cosine Similarity:**
   - Mengukur kemiripan antar film dengan menghitung **cosine similarity** antar vektor TF-IDF dari masing-masing film.
   - Hasilnya berupa matriks kemiripan berukuran `(9708, 9708)`, di mana setiap nilai mencerminkan tingkat kemiripan antara dua film.

2. **Rekomendasi Film:**
   - Untuk setiap film input, diambil 5 film yang memiliki nilai cosine similarity tertinggi.
   - Hasil rekomendasi diurutkan berdasarkan tingkat kemiripan, dan tidak termasuk film input itu sendiri.


### 💡 Alasan Pemilihan Model:

- Genre adalah informasi yang **tersedia untuk semua film**, sehingga cocok untuk cold-start item (film yang belum memiliki rating).
- TF-IDF membantu menyeimbangkan genre umum (seperti *Drama*) dan genre yang lebih unik, sehingga rekomendasi lebih seimbang.
- Cosine similarity efektif dalam mengukur kemiripan antar vektor dalam ruang berdimensi tinggi.

#### Contoh Hasil Rekomendasi : 
**🎬 Rekomendasi yang Film Mirip dengan "Antitrust (2001)" dengan genre Crime|Drama|Thriller**
| No | Title                                      | Genres                  |
|----|--------------------------------------------|--------------------------|
| 1  | Transsiberian (2008)                       | Crime\|Drama\|Thriller   |
| 2  | Presumed Innocent (1990)                   | Crime\|Drama\|Thriller   |
| 3  | Out of Time (2003)                         | Crime\|Drama\|Thriller   |
| 4  | Taking of Pelham 1 2 3, The (2009)         | Crime\|Drama\|Thriller   |
| 5  | Before the Devil Knows You're Dead         | Crime\|Drama\|Thriller   |

Dari hasil rekomendasi di atas, terlihat bahwa rekomendasi film yang mirip dengan dengan Antitrust (2001) merupakan film yang genrenya Crime, Drama dan Thriller.

### 2️⃣ Collaborative Filtering

Collaborative Filtering merupakan pendekatan berbasis interaksi antar pengguna (user) dan item (movie), tanpa memperhatikan konten eksplisit seperti genre. Pendekatan ini berupaya menemukan pola dari kesamaan preferensi antar pengguna untuk memberikan rekomendasi.

#### 🧠 Arsitektur Model

Model dikembangkan menggunakan TensorFlow dan Keras, serta memanfaatkan pendekatan **embedding-based neural network**. Detail arsitekturnya sebagai berikut:

- **User & Movie Embedding Layer**  
  Masing-masing user dan movie dipetakan ke dalam ruang vektor berdimensi tetap (default `embedding_size=50`) yang dilatih selama training.

- **User & Movie Bias**  
  Bias disertakan untuk memperhitungkan preferensi umum dari pengguna dan popularitas film tertentu.

- **Hidden Layer**  
  Representasi gabungan dari user dan movie embedding dilewatkan ke **Dense Layer**, kemudian diikuti oleh **Batch Normalization**, aktivasi ReLU, dan **Dropout** untuk regularisasi.

- **Output Layer**  
  Layer akhir adalah satu neuron yang memberikan prediksi rating dalam bentuk angka kontinu antara 0 dan 1, kemudian ditambahkan dengan bias pengguna dan film.


#### ⚙️ Pelatihan Model

- **Loss Function**: Mean Squared Error (MSE)  
- **Optimizer**: Adam (learning rate = 0.001)  
- **Early Stopping**: Diterapkan untuk mencegah overfitting dengan memantau `val_loss`.  
- **Batch Size**: 64  
- **Epochs**: Hingga 30, tetapi dapat berhenti lebih awal jika tidak ada peningkatan validasi.


#### 🙋‍♂️ Contoh Hasil Rekomendasi: 

**Top-10 Rekomendasi untuk Pengguna ID: 269**

##### 🎞️ Film dengan Rating Tinggi oleh Pengguna
| No | Judul                          | Genre                                  |
|----|--------------------------------|----------------------------------------|
| 1  | Heat (1995)                    | Action\|Crime\|Thriller                 |
| 2  | Bed of Roses (1996)           | Drama\|Romance                         |
| 3  | Toy Story (1995)              | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 4  | Craft, The (1996)             | Drama\|Fantasy\|Horror\|Thriller       |
| 5  | Leaving Las Vegas (1995)      | Drama\|Romance                         |

##### 🏆 Top-10 Film Rekomendasi  
| No | Judul                                                                                                 | Genre                                         |
|----|-----------------------------------------------------------------------------------------------------|-----------------------------------------------|
| 1  | Lifeboat (1944)                                                                                      | Drama\|War                                    |
| 2  | Persona (1966)                                                                                       | Drama                                         |
| 3  | Shrek Forever After (a.k.a. Shrek: The Final Chapter) (2010)                                        | Adventure\|Animation\|Children\|Comedy\|Fantasy\|IMAX |
| 4  | Guess Who's Coming to Dinner (1967)                                                                  | Drama                                         |
| 5  | Trial, The (Procès, Le) (1962)                                                                       | Drama                                         |
| 6  | Captain Fantastic (2016)                                                                             | Drama                                         |
| 7  | Day of the Doctor, The (2013)                                                                        | Adventure\|Drama\|Sci-Fi                      |
| 8  | Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997) | Action\|Animation\|Drama\|Fantasy\|Sci-Fi     |
| 9  | Man Bites Dog (C'est arrivé près de chez vous) (1992)                                               | Comedy\|Crime\|Drama\|Thriller                 |
| 10 | Three Billboards Outside Ebbing, Missouri (2017)                                                    | Crime\|Drama                                  |

Dari output diatas, User 269 tampaknya menyukai film dengan genre Drama dan Romance, tetapi juga menikmati variasi genre lain seperti Action, Thriller, dan Animation/Children. Ini memberi gambaran preferensi user yang cukup beragam, dengan kecenderungan kuat pada Drama. Dari hasil rekomendasi diperoleh :

- Mayoritas rekomendasi film ini bergenre Drama, yang konsisten dengan film-film yang user beri rating tinggi.

- Ada juga beberapa film dengan tambahan genre lain seperti War, Crime, Comedy, Romance, Sci-Fi, dan Animation, yang mengindikasikan model mencoba memberi variasi sekaligus tetap relevan dengan preferensi user.

- Rekomendasi ini tampak cocok untuk user karena menyesuaikan genre favoritnya, sekaligus memperkenalkan film-film klasik dan modern yang berpotensi menarik bagi user.

🔚 Dengan pendekatan Collaborative Filtering berbasis deep learning ini, sistem rekomendasi mampu mengidentifikasi pola tersembunyi dari interaksi pengguna, bahkan untuk film yang sebelumnya belum pernah ditonton oleh pengguna yang bersangkutan.

---

### ⚖️ Kelebihan & Kekurangan

| Pendekatan              | Kelebihan                                                                 | Kekurangan                                                                              |
|-------------------------|---------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Content-Based Filtering | - Tidak butuh data pengguna lain (cold start OK)                          <br> - Cocok untuk pengguna baru                                                           | - Rekomendasi cenderung monoton karena berbasis kemiripan konten                          <br> - Tidak belajar dari preferensi pengguna lain                                      |
| Collaborative Filtering | - Dapat menemukan pola tersembunyi antar pengguna dan film               <br> - Lebih variatif karena hasil dipengaruhi interaksi banyak pengguna                  | - Membutuhkan banyak data rating historis                                               <br> - Tidak cocok untuk user/film baru tanpa interaksi (cold start problem)             |

---

## 5. 📈 Evaluation

### 📐Metrik Evaluasi Content-Based-Filtering
Evaluasi dilakukan menggunakan metrik **Precision**, yang mengukur seberapa banyak dari hasil rekomendasi yang benar-benar relevan.

#### Rumus Precision:
Precision = (Jumlah item relevan yang direkomendasikan) / (Jumlah total item yang direkomendasikan)

Dalam kasus ini, definisi **relevan** adalah film yang memiliki **minimal satu genre yang sama** dengan film input (*Antitrust (2001)*). Genre film target adalah: `Crime | Drama | Thriller`.

Maka, Precision = 5 / 5 = 1.00


✅ **Hasil Precision: 1.00 (100%)**  
Artinya, semua film yang direkomendasikan memiliki genre yang relevan dengan film target. Model berhasil memberikan hasil rekomendasi yang sangat relevan secara konten.


### 📐 Metrik Evaluasi Collaborative Filtering
Digunakan **Root Mean Squared Error (RMSE)** sebagai metrik evaluasi untuk model Collaborative Filtering berbasis Neural Network.

#### Formula:
**RMSE = √(1/n × Σ(yᵢ - ŷᵢ)²)**
- RMSE rendah mengindikasikan bahwa prediksi model mendekati nilai sebenarnya (semakin akurat).

### 📉 Evaluasi Performa Model Collaborative Filtering

![RMSE Grafik](https://github.com/gabriellayp/RecommedationSystem/blob/main/images/metrikmodel.png?raw=true)

Visualisasi di atas menunjukkan performa model pada data pelatihan dan pengujian yang berhenti karena early stopping di epoch ke-10:

- **RMSE pada data pelatihan** menurun secara signifikan dari awal hingga sekitar epoch ke-4, lalu stabil di sekitar 0.19. Ini menunjukkan bahwa model mampu belajar representasi rating dengan baik.
- **RMSE pada data pengujian** juga menurun hingga sekitar epoch ke-4, tetapi kemudian mengalami fluktuasi kecil. Meski begitu, nilainya tetap berada di kisaran rendah (sekitar 0.20), yang menunjukkan performa generalisasi yang cukup baik.
- **Tidak terdapat indikasi overfitting signifikan**, karena selisih RMSE antara data pelatihan dan pengujian cukup kecil dan tidak menunjukkan lonjakan tajam.

🟢 **Kesimpulan**: Model Collaborative Filtering yang digunakan dapat memberikan prediksi rating film yang cukup akurat dan stabil terhadap data baru.

---

## 6. ✅ Kesimpulan

Proyek ini berhasil membangun sistem rekomendasi film yang efektif dengan mengimplementasikan dua pendekatan utama: Content-Based Filtering dan Collaborative Filtering berbasis Deep Learning.

Dari sisi Content-Based Filtering, sistem mampu memberikan rekomendasi film yang serupa berdasarkan kesamaan genre. Ini berguna untuk pengguna baru (cold start) yang belum memiliki riwayat interaksi. Pendekatan ini memanfaatkan teknik TF-IDF Vectorization dan Cosine Similarity untuk mengukur kemiripan antar film. Hasilnya menunjukkan bahwa sistem dapat merekomendasikan film-film dengan genre yang sangat relevan terhadap film yang disukai pengguna.

Sementara itu, pendekatan Collaborative Filtering menggunakan embedding model neural network untuk mempelajari hubungan kompleks antara pengguna dan film berdasarkan data rating historis. Model ini menunjukkan performa yang baik dalam memprediksi preferensi pengguna secara personal, terbukti dari nilai RMSE yang rendah dan stabil pada data uji. Dengan arsitektur deep learning yang sederhana namun efektif (user/movie embedding, dense layers, dan dropout), sistem mampu menggeneralisasi dengan baik tanpa overfitting.

Dari perspektif bisnis, sistem ini menjawab dua pertanyaan utama:

- Bagaimana memberikan rekomendasi yang akurat untuk pengguna baru maupun lama?

- Bagaimana memanfaatkan data historis untuk meningkatkan personalisasi?

Dengan meningkatnya kualitas rekomendasi, sistem ini berpotensi meningkatkan user engagement, retensi pengguna, dan kepuasan dalam menjelajahi konten film, yang merupakan aspek krusial dalam platform hiburan digital seperti Netflix dan sejenisnya.

Secara keseluruhan, proyek ini membuktikan bahwa kombinasi pendekatan berbasis konten dan kolaboratif dapat menghasilkan sistem rekomendasi yang lebih komprehensif, akurat, dan relevan, serta memiliki dampak langsung terhadap pengalaman pengguna dan nilai bisnis platform.

---

## 📚 Resources
- TensorFlow Docs: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Scikit-learn Docs: [https://scikit-learn.org/](https://scikit-learn.org/)

---

> ✨ Proyek ini merupakan bagian dari submission program pembelajaran sistem rekomendasi berbasis machine learning.
