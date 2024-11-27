# **Laporan Proyek Machine Learning - Riski Pratama**

## **Domain Proyek**
**Latar Belakang**

Diabetes merupakan salah satu masalah kesehatan utama yang menjadi perhatian global. Hal ini disebabkan oleh komplikasi serius yang dapat ditimbulkan oleh diabetes, seperti penyakit jantung, kerusakan ginjal, kerusakan saraf, dan masalah penglihatan. Berdasarkan laporan World Health Organization (WHO), prevalensi diabetes terus meningkat setiap tahun, terutama di negara-negara berkembang yang memiliki pola hidup kurang sehat.

Diabetes dapat terjadi karena berbagai faktor risiko, termasuk riwayat keluarga, gaya hidup yang tidak aktif, obesitas, tekanan darah tinggi, dan kadar glukosa darah yang tidak terkendali. Penanganan diabetes secara manual melalui pemeriksaan medis rutin sering kali memakan waktu dan biaya yang besar, sehingga pendekatan alternatif yang lebih efisien sangat dibutuhkan.

Dalam konteks ini, algoritma machine learning menjadi salah satu solusi yang efektif. Dengan memanfaatkan data medis pasien, seperti kadar glukosa, tekanan darah, indeks massa tubuh, dan riwayat keluarga, algoritma ini dapat digunakan untuk mendeteksi risiko diabetes secara dini. Pendekatan ini memungkinkan tenaga medis untuk mengambil tindakan preventif sebelum komplikasi serius terjadi.

Penerapan machine learning di bidang kesehatan telah menunjukkan banyak keberhasilan, seperti dalam diagnosis penyakit kanker, deteksi penyakit kardiovaskular, dan analisis citra medis seperti CT Scan atau MRI. Dengan menggunakan algoritma machine learning, analisis data kesehatan yang kompleks dapat dilakukan dengan lebih cepat dan akurat.

Pada proyek ini, dikembangkan sebuah model prediktif untuk memprediksi risiko diabetes berdasarkan data medis pasien. Dengan adanya model ini, diharapkan dapat membantu rumah sakit dan tenaga medis dalam mendeteksi pasien yang berisiko terkena diabetes secara lebih dini, sehingga intervensi yang tepat dapat dilakukan untuk mencegah komplikasi.
Pada proyek ini, tujuan utama adalah untuk memprediksi risiko diabetes berdasarkan data medis dari **[Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)**. Penyakit diabetes dapat menimbulkan komplikasi serius jika tidak didiagnosis dan ditangani sejak dini. Dengan memanfaatkan machine learning, prediksi dapat dilakukan lebih cepat dan akurat.

**Referensi**:
- UCI Machine Learning Repository: Pima Indians Diabetes Database.
- National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).

---

## **Business Understanding**

### **Problem Statements**
1. Fitur apa saja yang paling memengaruhi risiko seseorang terkena diabetes?
2. Bagaimana memastikan data medis pasien memiliki kualitas yang baik untuk digunakan dalam model machine learning?
3. Seberapa akurat hasil klasifikasi diabetes dengan fitur yang tersedia dalam dataset?

### **Goals**
1. Mengidentifikasi fitur yang paling signifikan dalam memprediksi risiko diabetes.
2. Mengembangkan sistem yang mampu mengelola data kesehatan pasien dengan aman dan efisien, termasuk fitur medis yang relevan untuk prediksi diabetes.
3. Membuat model machine learning yang dapat memprediksi risiko diabetes dengan tingkat akurasi yang tinggi.

### **Solution Statements**
1. Menggunakan beberapa algoritma machine learning, seperti Logistic Regression, Random Forest, dan Support Vector Machine (SVM).
2. Menerapkan teknik hyperparameter tuning untuk meningkatkan akurasi model.
3. Melakukan evaluasi menggunakan metrik seperti akurasi, precision, recall, F1-score, dan ROC-AUC untuk menilai performa model.

---

## **Data Understanding**

Dataset yang digunakan adalah **[Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)**. dari UCI Machine Learning Repository. Dataset ini terdiri dari 768 data pasien dengan 9 fitur medis yang relevan untuk memprediksi risiko diabetes. Berikut adalah deskripsi variabel yang ada:

### **Deskripsi Variabel**
| Variabel                  | Deskripsi                                                                 |
|---------------------------|---------------------------------------------------------------------------|
| `Pregnancies`             | Jumlah kehamilan                                                         |
| `Glucose`                 | Kadar glukosa plasma (mg/dL)                                             |
| `BloodPressure`           | Tekanan darah diastolik (mm Hg)                                          |
| `SkinThickness`           | Ketebalan kulit (mm)                                                     |
| `Insulin`                 | Kadar insulin serum (mu U/ml)                                            |
| `BMI`                     | Indeks massa tubuh (kg/m²)                                               |
| `DiabetesPedigreeFunction`| Probabilitas diabetes berdasarkan riwayat keluarga                       |
| `Age`                     | Usia pasien                                                             |
| `Outcome`                 | Label target (1 = diabetes, 0 = tidak)                                   |

### **Hasil Statistik Data**
Berikut adalah statistik deskriptif untuk fitur numerik dalam dataset:

| Fitur               | Count | Mean    | Std    | Min  | 25%  | 50%  | 75%  | Max  |
|---------------------|-------|---------|--------|------|------|------|------|------|
| `Pregnancies`       | 768   | 3.85    | 3.37   | 0    | 1    | 3    | 6    | 17   |
| `Glucose`           | 768   | 120.89  | 31.97  | 0    | 99   | 117  | 140  | 199  |
| `BloodPressure`     | 768   | 69.11   | 19.36  | 0    | 62   | 72   | 80   | 122  |
| `SkinThickness`     | 768   | 20.54   | 16.12  | 0    | 0    | 23   | 32   | 99   |
| `Insulin`           | 768   | 79.80   | 115.24 | 0    | 0    | 30   | 127  | 846  |
| `BMI`               | 768   | 31.99   | 7.88   | 0    | 27.3 | 32   | 36.6 | 67.1 |
| `DiabetesPedigreeFunction` | 768 | 0.47 | 0.33   | 0.08 | 0.24 | 0.37 | 0.63 | 2.42 |
| `Age`               | 768   | 33.24   | 11.76  | 21   | 24   | 29   | 41   | 81   |

### **Visualisasi Distribusi Data**
**Distribusi Kelas Target:**
- Visualisasi menunjukkan ketidakseimbangan data pada kelas target (Outcome) dengan lebih banyak pasien tanpa diabetes dibandingkan yang memiliki diabetes.
**Korelasi Antar Fitur Numerik:**
- Korelasi dianalisis menggunakan heatmap untuk memahami hubungan antar fitur numerik.
**Distribusi Berdasarkan Outcome:**
- Distribusi usia (Age) dan kehamilan (Pregnancies) dianalisis berdasarkan kelas target (Outcome) untuk mendapatkan wawasan tambahan.

### **Analisis Univariate**
**Fitur Kategori**:
- `Pregnancies`: Distribusi data menunjukkan bahwa mayoritas pasien memiliki 0-6 kehamilan.
- `Outcome`: Distribusi target menunjukkan ketidakseimbangan kelas (lebih banyak data non-diabetes).

**Fitur Numerik**:
- Fitur `Glucose` memiliki korelasi tertinggi dengan label `Outcome`.
- Distribusi `BMI` menunjukkan bahwa mayoritas pasien memiliki indeks massa tubuh antara 25-35.

### **Analisis Multivariate**
![Heatmap Kolerasi](https://github.com/user-attachments/assets/4eddd518-f68b-4dcb-a2a3-57ec6190159e)

**Korelasi Fitur Numerik**:
- `Glucose` menunjukkan korelasi positif yang signifikan terhadap `Outcome`.
- Korelasi antara fitur lainnya relatif rendah.

**Korelasi Fitur Kategori**:
- `Pregnancies` menunjukkan pola bahwa pasien dengan lebih banyak kehamilan memiliki risiko diabetes lebih tinggi.
- Fitur `Age` menunjukkan bahwa pasien yang lebih tua cenderung memiliki risiko lebih besar terkena diabetes.
---

## **Data Preparation**

Pada bagian ini, dilakukan beberapa tahapan untuk mempersiapkan data sebelum pemodelan:

1. **Pengecekan Missing Values**:
   - Dataset diperiksa untuk missing values menggunakan fungsi `isnull()`.

2. **Penanganan Ketidakseimbangan Data**:
   - Ketidakseimbangan pada kelas target (`Outcome`) diatasi menggunakan teknik **SMOTE**.

3. **Pembagian Data**:
   - Dataset dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split`.

4. **Standarisasi Fitur**:
   - Fitur numerik distandarisasi menggunakan `StandardScaler` untuk meningkatkan performa algoritma machine learning.
---

## **Modeling**

Pada tahap ini, beberapa algoritma machine learning diterapkan untuk memprediksi diabetes.

### **1. Logistic Regression (Baseline Model)**
Logistic Regression digunakan sebagai baseline model dengan hasil akurasi awal sebesar [0.6948]. Model ini dipilih karena kesederhanaan dan efisiensinya dalam menyelesaikan masalah klasifikasi biner.

**Parameter utama**:
- `solver`: `lbfgs` digunakan untuk optimasi.
- `max_iter`: Ditetapkan untuk memastikan konvergensi model.

**Kelebihan**: Cepat, sederhana, mudah diinterpretasikan.  
**Kekurangan**: Tidak bekerja dengan baik pada dataset yang non-linier atau sangat kompleks.

### **2. Random Forest**
Random Forest dipilih sebagai model berikutnya untuk menangani kompleksitas yang lebih tinggi pada dataset ini. Algoritma ini mampu menangani outliers dan data yang non-linier, dengan hasil akurasi sebesar [0.7208].

**Parameter utama**:
- `n_estimators`: Jumlah pohon dalam hutan ditetapkan menjadi 100.
- `max_depth`: Batas kedalaman tiap pohon ditetapkan untuk mencegah overfitting.

### **3. Hyperparameter Tuning**
Proses **GridSearchCV** diterapkan untuk mengoptimalkan parameter terbaik seperti `n_estimators`, `max_depth`, dan `min_samples_split`. Proses ini meningkatkan akurasi model sebesar [0.7468].

## **Evaluation**

### **1. Evaluasi Model**

Beberapa algoritma machine learning diuji pada dataset ini, termasuk Logistic Regression, Random Forest, Support Vector Machine (SVM), Naive Bayes, dan Voting Classifier. Berikut hasil evaluasi model:

| Model                           | Accuracy | Precision | Recall  | F1-Score | ROC-AUC |
|---------------------------------|----------|-----------|---------|----------|---------|
| Tuned Logistic Regression       | 0.6948   | 0.5556    | 0.7273  | 0.6299   | 0.8143  |
| Random Forest (Sebelum Tuning)  | 0.7208   | 0.6071    | 0.6182  | 0.6126   | 0.8125  |
| SVM                             | 0.7338   | 0.6458    | 0.5636  | 0.6019   | 0.8051  |
| Naive Bayes                     | 0.7662   | 0.6610    | 0.7091  | 0.6842   | 0.8253  |
| Voting Classifier               | 0.7532   | 0.6491    | 0.6727  | 0.6607   | 0.8187  |
| Tuned Random Forest (Setelah Tuning) | 0.7468 | 0.6429    | 0.6545  | 0.6486   | 0.8310  |

- **Confusion Matrix**: Setiap model divisualisasikan dengan Confusion Matrix untuk melihat distribusi prediksi benar dan salah.
- **ROC Curve**: Model dengan performa terbaik (Random Forest) dievaluasi lebih lanjut dengan ROC Curve untuk mengukur kemampuan membedakan kelas positif dan negatif.

### **2. Dampak Bisnis**

Evaluasi menunjukkan bahwa model **Random Forest** setelah tuning memiliki performa terbaik dengan ROC-AUC sebesar 0.8310. Hal ini penting karena:
- **Efisiensi Diagnosis**: Dengan akurasi tinggi, model dapat mendeteksi risiko diabetes lebih cepat, memungkinkan tindakan preventif dilakukan sebelum komplikasi muncul.
- **Pencegahan Biaya Medis**: Diagnosis dini membantu pasien menghindari perawatan medis yang lebih mahal akibat komplikasi.
- **Kesesuaian Goals**: Model ini berhasil menjawab problem statement dengan mengidentifikasi fitur signifikan (`Glucose` dan `BMI`), serta menunjukkan akurasi tinggi pada dataset yang telah diproses.
"""

Evaluasi model dilakukan menggunakan beberapa metrik:
![Confusion Matrix - Tuned Random Forest)](https://github.com/user-attachments/assets/5cf7dd83-0f34-45b1-9cd2-021e1b8323db)


### **3. Confusion Matrix**:  
Menampilkan distribusi prediksi benar dan salah antara kelas positif dan negatif.
![Confusion Matrix - XGBoost](https://github.com/user-attachments/assets/aefed641-614f-4970-adf4-8df3a3f44873)
![Confusion Matrix - Random Forest with SMOTE](https://github.com/user-attachments/assets/473d2e2f-fdcb-43b0-ba31-8f23fcb34312)

### **4. ROC Curve dan AUC Score**
ROC curve digunakan untuk mengevaluasi performa model secara keseluruhan pada berbagai threshold, sedangkan AUC (Area Under the Curve) mengukur seberapa baik model membedakan antara kelas positif dan negatif.
![ROC Curve dan AUC Score)](https://github.com/user-attachments/assets/d2f27f30-bcd9-43a7-a77f-e83828e6f9e5)


**Kesimpulan**:  
Model Random Forest menunjukkan performa terbaik dengan akurasi dan metrik evaluasi yang lebih tinggi dibandingkan Logistic Regression. Model ini mampu menangani kompleksitas dataset medis dan menghasilkan prediksi yang lebih akurat.

---


