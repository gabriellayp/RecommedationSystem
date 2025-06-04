# 🎬 Film Recommendation System
COHORT ID : MC008D5X2466

Nama : Gabriella Yoanda Pelawi

Email : mc008d5x2466@student.devacademy.id

## 1. 📌 Project Overview

Dalam era digital saat ini, industri hiburan semakin bergantung pada sistem rekomendasi untuk meningkatkan keterlibatan pengguna. Netflix mengklaim bahwa lebih dari 80% jam tayang berasal dari sistem rekomendasi mereka, menandakan betapa krusialnya sistem ini dalam pengalaman pengguna. 

Meningkatnya jumlah konten membuat pengguna kesulitan dalam memilih film yang relevan dengan preferensi mereka. Oleh karena itu, proyek ini bertujuan untuk membangun sistem rekomendasi film yang mampu memberikan rekomendasi yang relevan bagi pengguna berdasarkan riwayat interaksi dan kesamaan konten. Proyek ini penting karena mampu meningkatkan user engagement dan pengalaman pengguna dalam menjelajahi film. Oleh karena itu, kemampuan untuk membangun model rekomendasi yang akurat dan relevan dapat berdampak signifikan secara bisnis.

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
Dataset yang digunakan berasal dari Kaggle dengan judul [Movie Lens Small Latest Datset](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset/data)

### 🔢 Jumlah & Struktur Data
- **movies.csv:** 9,708 data film, dengan kolom `movieId`, `title`, dan `genres`.
- **ratings.csv:** 100,836 data rating oleh pengguna, dengan kolom `userId`, `movieId`, `rating`, dan `timestamp`.

### 🔍 Fitur
| Fitur       | Deskripsi                                     |
|-------------|-----------------------------------------------|
| `movieId`   | ID unik film                                  |
| `title`     | Judul film                                    |
| `genres`    | Genre film dalam format string (dipisah '|')  |
| `userId`    | ID unik pengguna                              |
| `rating`    | Nilai rating dari pengguna terhadap film      |
| `timestamp` | Waktu saat pengguna memberikan rating         |

### 📊 Exploratory Data Analysis
**1. Jumlah Data Variabel**
| Fitur       | Jumlah                                     |
|-------------|-----------------------------------------------|
| `userId`   | 610                                  |
| `movieId`     | 9724                                    |
| `rating`    | 100,836  |

**2. Distribusi Genre Film 🎬**
   ![Distribusi Genre](https://github.com/gabriellayp/RecommedationSystem/blob/main/images/Genrefilm.png?raw=true)
   Genre yang paling mendominasi dalam dataset adalah Drama, diikuti oleh Comedy dan Thriller. Hal ini menunjukkan bahwa film dengan genre drama adalah yang paling sering diproduksi atau tersedia dalam data yang digunakan. Genre-genre seperti IMAX, Film-Noir, dan Western memiliki jumlah film yang jauh lebih sedikit, menunjukkan bahwa film dengan genre tersebut lebih jarang muncul. Distribusi ini dapat memengaruhi performa model, terutama pada sistem rekomendasi berbasis genre seperti content-based filtering, karena genre dominan akan lebih sering direkomendasikan.

**3. Distribusi Rating Film ⭐**
  ![Distribusi Rating](https://github.com/gabriellayp/RecommedationSystem/blob/main/images/ratingfilm.png?raw=true)
Distribusi rating menunjukkan bahwa sebagian besar pengguna memberikan rating yang cukup tinggi terhadap film yang mereka tonton. Rating paling umum berada di angka 4.0, diikuti oleh 3.0 dan 5.0. Sebaliknya, rating rendah (seperti 0.5–1.5) jauh lebih jarang diberikan.Hal ini menunjukkan bahwa pengguna cenderung lebih sering menonton dan menilai film yang mereka sukai, atau film dengan kualitas yang relatif baik. Pola ini penting untuk diperhatikan karena bisa menciptakan bias pada model rekomendasi, terutama pada pendekatan collaborative filtering.

---

## 4. 🛠️ Data Preparation

### ✅ Langkah-Langkah
1. **Menghapus label tidak valid:**  Label `(no genres listed)` pada kolom genre dihapus karena tidak memberikan informasi yang berguna untuk analisis.
2. **Merge Data:** Menggabungkan dataset movies.csv dan ratings.csv berdasarkan kolom movieId untuk mengaitkan setiap film dengan rating  yang diberikan pengguna. Hasil penggabungan ini menghasilkan data frame dengan kolom `userId`, `movieId`, `rating`, `timestamp`, `title`, dan `genres`.
3. **Pivot Table:** Data diubah menjadi format matriks dengan judul film (`title`) sebagai indeks dan ID pengguna (`userId`) sebagai kolom. 
4. **Missing Values:** Karena tidak semua pengguna memberi rating pada semua film, maka akan muncul nilai NaN. Nilai tersebut diasumsikan sebagai belum dirating dan diisi dengan 0
5. **Encoding:** Untuk keperluan pemodelan (terutama embedding), seperti pada Neural Collaborative Filtering (NCF) atau Matrix Factorization, baik `userId` maupun `movieId` perlu diubah menjadi representasi numerik berupa indeks integer mulai dari 0 hingga N-1.
6. **Filtering:** Data disaring untuk menyertakan hanya film yang memiliki minimal 5 rating dari pengguna berbeda dan Pengguna yang memberikan minimal 5 rating ke film yang berbeda.
Tujuannya adalah menjaga kualitas data agar fokus hanya pada entitas yang aktif dan relevan.

Note : Fitur timestamp tidak digunakan dalam analisis karena tidak relevan terhadap tujuan pemodelan (content-based recommendation dan colaborative filtering).

### ⚙️ Alasan
- Merge data bertujuan menyatukan informasi film dan interaksi pengguna ke dalam satu frame kerja analisis.
- Pivot table dibutuhkan untuk mengubah data transaksional ke bentuk matriks interaksi user-item yang siap untuk digunakan dalam sistem rekomendasi.
- Imputasi nilai kosong dengan 0 adalah pendekatan umum dalam explicit feedback dataset, mengasumsikan bahwa ketiadaan interaksi berarti ketidaktertarikan atau ketidaktahuan.
- Encoding digunakan agar input yang sebelumnya kategorikal dapat diproses oleh model numerik, termasuk untuk embedding dan matrix factorization.
- Filtering membantu mengurangi noise, mempercepat pelatihan model, dan meningkatkan kualitas rekomendasi dengan hanya mempertahankan entitas yang aktif dan informatif.

---

## 🤖 Modeling and Results

### 1️⃣ Content-Based Filtering (Genre-Based)

Content-Based Filtering adalah pendekatan rekomendasi yang berfokus pada kemiripan konten antar item. Dalam proyek ini, konten yang dimaksud adalah **genre film**, dan kemiripan antar film dihitung berdasarkan representasi numerik dari genre tersebut menggunakan teknik **TF-IDF (Term Frequency–Inverse Document Frequency)**.


### 📌 Langkah-Langkah:

1. **TF-IDF Vectorization:**
   - Genre film dalam kolom `genres` diubah menjadi vektor numerik menggunakan `TfidfVectorizer`.
   - Karena genre dipisahkan oleh simbol `|`, digunakan `token_pattern=r'[^|]+'` agar setiap genre dianggap sebagai satu token.
   - Hasil akhir berupa matriks TF-IDF berukuran `(9708, 19)`—yang berarti terdapat 9708 film dan 19 genre unik.

2. **Cosine Similarity:**
   - Mengukur kemiripan antar film dengan menghitung **cosine similarity** antar vektor TF-IDF dari masing-masing film.
   - Hasilnya berupa matriks kemiripan berukuran `(9708, 9708)`, di mana setiap nilai mencerminkan tingkat kemiripan antara dua film.

3. **Rekomendasi Film:**
   - Untuk setiap film input, diambil 5 film yang memiliki nilai cosine similarity tertinggi.
   - Hasil rekomendasi diurutkan berdasarkan tingkat kemiripan, dan tidak termasuk film input itu sendiri.


### 💡 Alasan Pemilihan Model:

- Genre adalah informasi yang **tersedia untuk semua film**, sehingga cocok untuk cold-start item (film yang belum memiliki rating).
- TF-IDF membantu menyeimbangkan genre umum (seperti *Drama*) dan genre yang lebih unik, sehingga rekomendasi lebih seimbang.
- Cosine similarity efektif dalam mengukur kemiripan antar vektor dalam ruang berdimensi tinggi.

#### Contoh Hasil Rekomendasi : 
**🎬 Rekomendasi yang Film Mirip dengan "Antitrust (2001)"**
| No | Title                                      | Genres                  |
|----|--------------------------------------------|--------------------------|
| 1  | Transsiberian (2008)                       | Crime\|Drama\|Thriller   |
| 2  | Presumed Innocent (1990)                   | Crime\|Drama\|Thriller   |
| 3  | Out of Time (2003)                         | Crime\|Drama\|Thriller   |
| 4  | Taking of Pelham 1 2 3, The (2009)         | Crime\|Drama\|Thriller   |
| 5  | Before the Devil Knows You're Dead         | Crime\|Drama\|Thriller   |


### 2️⃣ Collaborative Filtering

Collaborative Filtering merupakan pendekatan berbasis interaksi antar pengguna (user) dan item (movie), tanpa memperhatikan konten eksplisit seperti genre. Pendekatan ini berupaya menemukan pola dari kesamaan preferensi antar pengguna untuk memberikan rekomendasi.

#### ✅ Tahapan Pembangunan Model:

1. **Transformasi dan Pembersihan Data**  
   Dataset yang awalnya dalam format pivot (user vs movie rating) diubah menjadi format panjang (long format). Semua entri dengan rating nol dibuang karena dianggap sebagai ketidakterlibatan pengguna.

2. **Filtering Data**  
   Untuk meningkatkan kualitas rekomendasi dan efisiensi model, hanya pengguna dan film dengan minimal **5 interaksi** yang disertakan. Hal ini penting untuk menghindari cold-start dan memastikan model cukup data untuk belajar.

3. **Encoding**  
   User dan title diubah menjadi representasi numerik menggunakan teknik mapping (`user_to_user_encoded`, `movie_to_movie_encoded`) karena input model neural network harus berupa angka.

4. **Normalisasi Rating**  
   Nilai rating dinormalisasi ke rentang 0–1 untuk mempercepat dan menstabilkan proses pelatihan, khususnya ketika menggunakan fungsi aktivasi dan optimisasi berbasis gradien.

5. **Pembagian Data**  
   Data dibagi menjadi training dan validation set menggunakan **GroupShuffleSplit** berdasarkan `user`, untuk menghindari kebocoran informasi antar grup pengguna.


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

🔚 Dengan pendekatan Collaborative Filtering berbasis deep learning ini, sistem rekomendasi mampu mengidentifikasi pola tersembunyi dari interaksi pengguna, bahkan untuk film yang sebelumnya belum pernah ditonton oleh pengguna yang bersangkutan.

---

### ⚖️ Kelebihan & Kekurangan

| Pendekatan              | Kelebihan                                                                 | Kekurangan                                                                              |
|-------------------------|---------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Content-Based Filtering | - Tidak butuh data pengguna lain (cold start OK)                          <br> - Cocok untuk pengguna baru                                                           | - Rekomendasi cenderung monoton karena berbasis kemiripan konten                          <br> - Tidak belajar dari preferensi pengguna lain                                      |
| Collaborative Filtering | - Dapat menemukan pola tersembunyi antar pengguna dan film               <br> - Lebih variatif karena hasil dipengaruhi interaksi banyak pengguna                  | - Membutuhkan banyak data rating historis                                               <br> - Tidak cocok untuk user/film baru tanpa interaksi (cold start problem)             |

---

## 5. 📈 Evaluation

### 📐 Metrik Evaluasi
Kami menggunakan **Root Mean Squared Error (RMSE)** sebagai metrik evaluasi untuk model Collaborative Filtering berbasis Neural Network.

#### Formula:
**RMSE = √(1/n × Σ(yᵢ - ŷᵢ)²)**
- RMSE rendah mengindikasikan bahwa prediksi model mendekati nilai sebenarnya (semakin akurat).

### 📉 Evaluasi Performa Model Collaborative Filtering

![RMSE Grafik](attachment/fd563d88-b942-4686-b1bd-abcf294f10a6.png)

Visualisasi di atas menunjukkan performa model pada data pelatihan dan pengujian selama 10 epoch:

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
