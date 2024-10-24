FİSK DATASET PROJECT SUMMARY
İlk olarak, veri analizi, görselleştirme ve makine öğrenimi için gerekli kütüphaneler olan Pandas, NumPy, Matplotlib, Scikit-learn, OS, Struct, Warnings, TensorFlow, Pillow ve Keras'ı içe aktardım. Ayrıca, uyarıları göz ardı etmek için bir ayar yaptım. 
Sonra , balık görüntülerinin bulunduğu dizindeki tüm .png dosyalarını bulup, her birinin etiketini ve dosya yolunu çıkardım. Bu bilgileri iki listeye (label ve path) ekledim. Daha sonra, bu listeleri kullanarak bir Pandas DataFrame oluşturdum ve path ile label sütunlarını doldurdum. Ayrıca, "GT" (Ground Truth) klasörlerini göz ardı ettim. 
Verileri inceleme aşamasında ilk olarak, , oluşturduğum DataFrame'in ilk beş satırını ve son beş satırını görüntüledim. Bu, veri yapısını ve içeriğini kontrol etmeme yardımcı oldu. df.head() ve df.tail() komutu, path ve label sütunlarıyla birlikte ilk birkaç örneği ve son birkaç göstererek veri setinin doğru bir şekilde yüklendiğini doğrulamamı sağladı.  Ayrıca burada df.shape komutunu kullanarak DataFrame'in boyutunu kontrol ettim. Bu, toplam kayıt sayısını ve sütun sayısını gösterir. Böylece veri setimin büyüklüğünü ve yapısını daha iyi anlayarak, analiz ve modelleme aşamalarında hangi verilerle çalışacağımı belirlemiş oldum.
 Burada ayrıca, print(df) komutunu kullanarak DataFrame'in tamamını ekrana yazdırdım. Bu, tüm veri setini gözlemlememi sağladı ve path ile label sütunlarındaki verilerin doğruluğunu kontrol etmeme yardımcı oldu. 
 Böylece, veri setinin içeriği hakkında genel bir bakış elde ettim. 
 Sonrasında , df['label'].unique() komutunu kullanarak veri setindeki benzersiz etiketleri çıkardım. Ardından, her bir etiket için bir örnek görüntüyü gösteren 3x3 bir alt grafik oluşturmak üzere plt.subplots kullandım. Her benzersiz etiket için ilk görüntüyü okuyup alt grafik üzerinde görüntüledim ve başlık olarak etiket adını ekledim. Bu, her türün temsilini görselleştirerek modelin öğrenmesi gereken sınıfları anlamama yardımcı oldu. 
 Daha sonra , görüntüleri yüklemek ve ön işlemek için bir fonksiyon (load_images) tanımladım.
Fonksiyonun İşlevi:DataFrame'deki her bir görüntü dosyasını açıp belirli bir boyuta (img_size) yeniden boyutlandırdım.
Görüntüleri normalleştirerek (0-1 aralığına ölçekleyerek) modelin eğitimine uygun hale getirdim.
Her bir görüntüyü ve karşılık gelen etiketini listelere ekledim.
Çıktı: Fonksiyon, işlenmiş görüntülerin ve etiketlerin NumPy dizileri olarak döndürülmesini sağlıyor. Bu, daha sonra model eğitiminde kullanılmak üzere veri setini hazır hale getiriyor.

Sonra , veri setini eğitim ve test setlerine ayırdım.
Fonksiyon Kullanımı: train_test_split fonksiyonunu kullanarak veri setimin %80'ini eğitim, %20'sini test için ayırdım.
test_size: Test setinin boyutunu belirliyor (bu durumda %20).
random_state: Sonuçların tekrarlanabilir olmasını sağlamak için bir rastgelelik tohum değeri ayarladım.
stratify: Etiketlerin dağılımının eğitim ve test setlerinde benzer olmasını sağlamak için kullanıldı, böylece her iki set de her etiket için benzer oranlara sahip olur.
Sonuç: Eğitim ve test setlerinin içeriğini yazdırarak ayırmanın başarılı olup olmadığını kontrol ettim. Bu, modelin daha iyi genelleştirilmesi için kritik bir adımdır. 

Daha sonra, eğitim ve test setlerindeki görüntüleri yükledim ve etiketleri one-hot encoding formatına dönüştürdüm.
Görsellerin Yüklenmesi:load_images fonksiyonunu kullanarak, X_train_df ve X_test_df DataFrame'lerinden görüntüleri ve etiketleri yükledim. X_train ve X_test olarak görüntüleri, y_train ve y_test olarak etiketleri elde ettim.
Etiketlerin One-Hot Encoding'e Dönüştürülmesi:Öncelikle, benzersiz etiketleri çıkardım ve her bir etiket için bir indeks belirledim (sınıf haritası oluşturma).
y_train ve y_test dizilerini, bu haritayı kullanarak sayısal değerlere dönüştürdüm.
to_categorical fonksiyonunu kullanarak etiketleri one-hot encoding formatına dönüştürdüm. Bu, modelin etiketleri daha etkili bir şekilde öğrenmesine yardımcı olur.
Sonuç: Eğitim ve test setlerinin boyutlarını yazdırarak, yüklenen görüntülerin ve dönüştürülen etiketlerin doğru bir şekilde hazırlandığını kontrol ettim.

Modeli eğitme aşamasında, bir yapay sinir ağı modeli oluşturdum ve derledim.

Model Oluşturma:Sequential sınıfını kullanarak katmanları sırayla eklediğim bir model oluşturdum.
İlk katman olarak Flatten kullandım; bu, 64x64 boyutundaki görüntüleri düzleştirerek bir vektör haline getiriyor.
İki adet gizli katman ekledim:
İlk katman: 128 nöron ve relu aktivasyon fonksiyonu.
İkinci katman: 64 nöron ve relu aktivasyon fonksiyonu.
Son katman: len(classes) kadar nöron, softmax aktivasyon fonksiyonu ile çıkış veriyor. Bu, sınıflar arasındaki olasılık dağılımını sağlıyor. 

Modeli Derleme:adam optimizasyon algoritmasını kullandım.
Kayıp fonksiyonu olarak categorical_crossentropy seçtim, bu da çoklu sınıf sınıflandırma problemleri için uygun.
Modelin performansını değerlendirmek için accuracy metriğini ekledim.
Bu adımda, modelim eğitim ve test aşamalarına hazır hale getirildi.

Eğitim sürecinin sonunda, modelin öğrenme sürecini ve performansını takip edebilmek için history nesnesinde eğitim ve doğrulama kayıplarını ve doğruluklarını sakladım. Bu, modelin gelişimini analiz etmemi sağlayacak. 

Değerlendirme aşamasında ,  eğitimini tamamladığım modelin test seti üzerindeki performansını değerlendirdim.

Modelin Değerlendirilmesi:
evaluate fonksiyonunu kullanarak, test verileri (X_test ve y_test) ile modelin kaybını (test_loss) ve doğruluğunu (test_acc) hesapladım.
Sonuç:
Test setindeki doğruluk değerini ekrana yazdırdım. Bu, modelin daha önce görmediği verilerle ne kadar iyi performans gösterdiğini anlamamı sağladı. Yüksek bir doğruluk, modelin genel başarısını gösterirken, düşük bir doğruluk modelin geliştirilmesi gerektiğini işaret eder.

Son olarak,  modelin eğitim sürecini ve performansını görselleştirdim.

Doğruluk Grafiği:

plt.plot fonksiyonu ile eğitim ve doğrulama doğruluğunu çizdim.
history.history['accuracy'] ile eğitim doğruluğunu, history.history['val_accuracy'] ile doğrulama doğruluğunu gösterdim.
X eksenine epok sayısını, Y eksenine doğruluk değerini ekledim.
Grafik Detayları: 
Grafikte iki farklı çizgi ile eğitim ve doğrulama doğruluğunu temsil ettim.
plt.legend() ile çizgilerin etiketlerini ekledim ve plt.show() ile grafiği görüntüledim.
Bu görselleştirme, modelin eğitim sürecindeki ilerlemesini ve potansiyel aşırı öğrenme (overfitting) durumlarını anlamama yardımcı oldu. Eğitim ve doğrulama doğrulukları arasındaki farkı gözlemleyerek modelin performansını değerlendirdim. 

KAGGLE LİNKİ: https://www.kaggle.com/code/ismailerennn/fish-dataset-project
