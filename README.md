# 📞 Bank Telemarketing Prediction App

Bu proje, bir bankanın tele-pazarlama kampanyaları kapsamında müşterilerin vadeli mevduat hesabı açıp açmayacağını (abone olup olmayacağını) tahmin eden uçtan uca bir makine öğrenmesi uygulamasıdır.

## 🚀 Proje Hakkında

Uygulama, geçmiş kampanya verilerinden öğrenilen örüntüleri kullanarak yeni müşteri adayları için öngörüler oluşturur. Karmaşık veri ön işleme adımları (Cyclical Encoding, Feature Engineering) bir **Scikit-Learn Pipeline** yapısı içinde toplanmış ve **Streamlit** ile kullanıcı dostu bir arayüzle sunulmuştur.

### Temel Özellikler:

* **Özel Veri Mühendisliği:** Ay ve gün verileri için sinüs/kosinüs dönüşümü (cyclical encoding) yapılarak zamanın döngüsel doğası korunmuştur.
* **Pipeline Entegrasyonu:** Tüm dönüşümler ve model tek bir dosya (`model.joblib`) içinde paketlenmiştir.
* **İnteraktif Arayüz:** Kullanıcıdan alınan verilerle gerçek zamanlı olasılık tahmini ve sınıflandırma yapılır.

---

## 🛠️ Kullanılan Teknolojiler

* **Python 3.x**
* **Streamlit:** Web arayüzü geliştirme.
* **Scikit-Learn:** Model kurma ve pipeline mimarisi.
* **Pandas & NumPy:** Veri manipülasyonu ve matematiksel dönüşümler.
* **Joblib:** Modelin kaydedilmesi ve yüklenmesi.

---

## 🏗️ Mimari ve Veri Ön İşleme

Proje içerisinde yer alan `BankFeatureEngineer` sınıfı şu işlemleri otomatik olarak gerçekleştirir:

1. **Binary Mapping:** 'Yes/No' değerlerini `1/0` formatına dönüştürür.
2. **Feature Creation:** Bakiyenin negatif olup olmaması (`is_non_negative_balance`) ve müşterinin yeni olup olmaması (`new_client`) gibi yeni öznitelikler türetir.
3. **Cyclical Encoding:** `month` ve `day` sütunlarını periyodik fonksiyonlara (sin/cos) dönüştürerek modelin takvim etkisini anlamasını sağlar.
4. **Category Dtype:** Kategorik sütunları Gradient Boosting modellerine uygun şekilde optimize eder.

---

## 💻 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için şu adımları izleyin:

1. **Depoyu Klonlayın:**
```bash
git clone https://github.com/kullaniciadin/bank-telemarketing-prediction.git
cd bank-telemarketing-prediction

```


2. **Gerekli Kütüphaneleri Yükleyin:**
```bash
pip install streamlit pandas numpy scikit-learn 

```

ya da

```bash
pip install -r requirements.txt

```


3. **Uygulamayı Başlatın:**
```bash
streamlit run app.py

```

---

## 📊 Örnek Kullanım

1. Sol panelden veya ana ekrandan müşterinin **Yaş, Meslek, Eğitim** gibi demografik bilgilerini girin.
2. **Finansal durum** (Bakiye, Kredi borcu vb.) bilgilerini doldurun.
3. **"🔮 Tahmin Et"** butonuna basın.
4. Uygulama size müşterinin abone olma **olasılığını (%)** ve **nihai kararını** (Abone Olur/Olmaz) anında gösterecektir.

---

## 📁 Dosya Yapısı

* `app.py`: Streamlit arayüzü ve model yükleme mantığı.
* `model.joblib`: Eğitilmiş Scikit-Learn pipeline nesnesi.
* `README.md`: Proje dökümantasyonu.

---

### 💡 Not

Modelin doğru çalışabilmesi için `BankFeatureEngineer` sınıf tanımının `app.py` içinde yer alması kritik önem taşır. `joblib` nesneyi yüklerken bu sınıfın şablonuna ihtiyaç duyar.

---


## Veri Seti Kaynağı

- Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.
- https://archive.ics.uci.edu/dataset/222/bank+marketing