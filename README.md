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
git clone <https://github.com/nailBestas/llm_zero_to_hero_experiments.git>
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
README’ye ekleyebileceğin kısa ve net bir bölüm şöyle olabilir.

***

## Proje Özeti

Bu repo, sıfırdan Mini Transformer tabanlı bir LLM kurup eğitmenin yanında, GeeksforGeeks’in tanımladığı 10 ana yapay zeka dalının her biri için çalışan küçük demolar içerir. Amaç, hobi ve öğrenme odaklı tek bir proje içinde AI ekosisteminin geniş bir kısmına dokunmaktır.

## AI Dalları ve Modüller

| AI dalı                     | Dosya/Yol                                                       | Nasıl çalıştırılır (proje kökünde)                                             |
|----------------------------|------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Machine Learning (ML)      | `src/ai_domains/classical_ml/train_classic_ml.py`               | `PYTHONPATH=. python3 src/ai_domains/classical_ml/train_classic_ml.py`         |
| Deep Learning / LLM        | `src/transformer_core/*`, `src/train/train_minilm.py`           | `PYTHONPATH=. python3 src/train/train_minilm.py`                                |
| Natural Language Processing| LLM + inference/API kodları                                     | Örn. `PYTHONPATH=. python3 scripts/generate_demo.py --prompt "hello world"`    |
| Computer Vision            | `src/ai_domains/vision/vision_demo.py`                          | `PYTHONPATH=. python3 src/ai_domains/vision/vision_demo.py --images-dir images`|
| Reinforcement Learning     | `src/ai_domains/rl/rl_cartpole.py`                              | `PYTHONPATH=. python3 src/ai_domains/rl/rl_cartpole.py --timesteps 10000 --episodes 5` |
| Expert Systems             | `src/ai_domains/expert_systems/rule_engine.py`                  | `PYTHONPATH=. python3 src/ai_domains/expert_systems/rule_engine.py --interactive` |
| Search & Planning          | `src/ai_domains/planning_search/search_algos.py`                | `PYTHONPATH=. python3 src/ai_domains/planning_search/search_algos.py`          |
| Fuzzy Logic                | `src/ai_domains/fuzzy_logic/fuzzy_controller.py`                | `PYTHONPATH=. python3 src/ai_domains/fuzzy_logic/fuzzy_controller.py --temp 22`|
| Evolutionary Computation   | `src/ai_domains/evolutionary/genetic_algorithm_demo.py`         | `PYTHONPATH=. python3 src/ai_domains/evolutionary/genetic_algorithm_demo.py --generations 40` |
| Swarm Intelligence         | `src/ai_domains/swarm_intelligence/pso_demo.py`                 | `PYTHONPATH=. python3 src/ai_domains/swarm_intelligence/pso_demo.py --iterations 40` |

Bu yapı sayesinde tek bir repo içinde: klasik ML, derin öğrenme/LLM, NLP, bilgisayarla görü, RL, uzman sistemler, arama/planlama, bulanık mantık, evrimsel algoritmalar ve sürü zekâsı için uçtan uca çalışan örnekler bulunur.

***

🤝 Katkıda Bulunma

Pull request, issue veya iyileştirme önerilerin memnuniyetle karşılanır.
