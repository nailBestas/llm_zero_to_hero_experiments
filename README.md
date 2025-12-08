llm_zero_to_hero_experiments

Bu proje, büyük dil modelleri (LLM) ile “sıfırdan ileri seviyeye” doğru deneyler yapmayı amaçlayan, düzenli klasör yapısına sahip, kolay genişletilebilir bir deney altyapısı sunar. Veri hazırlama, model testleri, deney betikleri, Docker desteği ve otomasyon yapısı bir arada bulunur.

🚀 Proje Amacı

LLM tabanlı deneyleri düzenli bir altyapı ile yürütmek

Farklı veri setleri, promptlar veya parametrelerle A/B karşılaştırmaları yapmak

Tekrarlanabilir deney ortamı oluşturmak

Docker, sanal ortam ve test desteği ile profesyonel bir çalışma düzeni sağlamak

📁 Klasör Yapısı
llm_zero_to_hero_experiments/
├── data/                  # Veri dosyaları
├── src/                   # Ana kaynak kod
├── scripts/               # Yardımcı komut/işleme betikleri
├── scripts_experiments/   # Deney betikleri
├── tests/                 # Birim testler
├── requirements.txt       # Python bağımlılıkları
├── Dockerfile             # Docker imajı için yapılandırma
└── README.md              # Proje dokümantasyonu

🛠️ Kurulum
1) Reponun klonlanması
git clone https://github.com/nailBestas/llm_zero_to_hero_experiments.git
cd llm_zero_to_hero_experiments

2) Sanal ortam (opsiyonel)
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\Scripts\activate     # Windows

3) Bağımlılıkların yüklenmesi
pip install -r requirements.txt

4) Docker ile çalışma (opsiyonel)
docker build -t llm-zero-to-hero .
docker run --rm -it llm-zero-to-hero bash

🧪 Deneyleri Çalıştırma

Tüm deney betikleri scripts_experiments klasöründedir.

Veri hazırlama, dönüştürme vb. işlemler scripts klasöründedir.

Model testleri ve doğrulamalar tests klasöründe bulunur.

İstersen kendi veri setini data/ klasörüne koyarak kolayca yeni deneyler oluşturabilirsin.

🎯 Kimler İçin?

LLM modelleriyle pratik deney yapmak isteyen geliştiriciler

Kendi veri setiyle model test etmek isteyen araştırmacılar

Prompt mühendisliği, parametre denemeleri, A/B karşılaştırmaları yapmak isteyen kullanıcılar

Tekrarlanabilir ve düzenli bir LLM deney ortamı arayan herkes

🤝 Katkıda Bulunma

Pull request, issue veya iyileştirme önerilerin memnuniyetle karşılanır.
