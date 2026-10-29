# Hane Alışveriş Asistanı

dunnhumby **Complete Journey** verisindeki 801 hanenin 2017 alışverişlerini **yaş, gelir, hane büyüklüğü, hane tipi ve çocuk
durumuna** göre analiz eden bir sohbet asistanı. "Yaşa göre ürün tercihi değişiyor mu?", "Çocuklu haneler hangi kategorilerde öne
çıkıyor?", "Gelir arttıkça özel marka tercihi değişiyor mu?" gibi Türkçe soruları yanıtlar.

Qwen3 8B (Ollama) hangi analizin çalışacağını **planlar**, hesabı Python (pandas + SciPy) yapar ve sonucu doğrulanmış Türkçe bulgu
cümlelerine çevirir; Qwen cevabı yalnızca bu cümlelerden yazar. Cevaptaki sayılar bulgularla karşılaştırılır, tutmazsa bir kez
yeniden yazdırılır, yine tutmazsa bulgular aynen gösterilir.

## Verinin sınırı (önce bunu okuyun)

- Demografi **801 hane** için bilinir; cinsiyet **yoktur** (asistan bu soruda uydurmaz, yokluğunu söyler).
- Analiz birimi **hane**dir. Bir hane yüzlerce kalem aldığı için kalemleri bağımsız saymak her farkı "anlamlı" gösterirdi;
  bu yüzden her test 801 hane üzerinden yapılır.
- Ürünler kategori düzeyindedir (298 kategori); marka yalnızca ulusal marka / özel marka ayrımıdır.
- Tek yıl (2017), tek ABD market zinciri; Türkiye'yi temsil etmez.
- Her demografik boyut ayrı test edilir; yaş, gelir ve çocuk durumu birbiriyle ilişkilidir.
- Küçük gruplar (örn. 19-24 yaş, 46 hane) için güç düşüktür; "fark yok" demek "fark olmadığı kanıtlandı" demek değildir.

## Çalıştırma

```bash
ollama pull qwen3:8b

cd backend
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn sales_assistant.api.app:create_app --factory --port 8000
```

İkinci terminalde:

```bash
cd frontend
npm install
npm run dev
```

Tarayıcıda http://localhost:5173. Ortam değişkenleri: `QWEN_MODEL`, `OLLAMA_URL`, `NUM_CTX`, `DATA_DIR`, `FEEDBACK_DB`.

## Veri hazırlama

`backend/data/` içindeki iki tablo hazır gelir. Yeniden üretmek için `completejourney-py` ile dışa aktarılan klasörden:

```bash
python -m sales_assistant.data.build --raw ../cj_data
```

| Tablo | İçerik |
|---|---|
| `households.csv.gz` | 801 satır: demografi + yıllık harcama, sepet sayısı, ortalama sepet, kategori çeşidi, özel marka payı |
| `purchases.csv.gz` | hane × kategori yıllık harcaması (99.685 satır) |

Betik demografisi olan hanelerin işlemlerini alır, sıfır/negatif tutarları ve kupon satırlarını çıkarır, geliri 12 sınıftan 4 gruba
indirir (küçük sınıflar tek başına analiz edilemez).

## Mimari

```
backend/sales_assistant/
  settings.py            ortam değişkenlerinden ayarlar
  container.py           bağımlılıkların kurulumu
  service.py             sohbeti kaydeden ve hataları olaya çeviren katman
  domain/                demografik boyutlar, sınırlamalar, metin normalleştirme
  data/                  build.py (hazırlık), repository.py (yükleme), resolver.py (isim eşleştirme)
  analytics/             evidence.py (istatistik), registry.py (araç kaydı), tools.py (6 araç), narration.py, charts.py
  llm/                   ollama.py, prompts.py, planner.py, grounding.py, pipeline.py
  feedback/store.py      SQLite: beğeni, neden, düzeltme ve sft/dpo/eval dışa aktarımı
  api/app.py             Starlette: /api/chat (SSE), /api/feedback, /api/health, /api/export/{kind}
  evaluation/            planlayıcı ve dürüstlük değerlendirmesi
backend/tests/           unittest
frontend/                React + Vite + Recharts
```

Akış: soru → **planlayıcı** (Qwen, JSON şeması zorlanır, sıcaklık 0) → **araçlar** (Python; hesaplar ve bulgu cümleleri üretir) →
**yanıtlayıcı** (Qwen yalnızca bulguları görür) → **denetim** (sayılar ve desteklenmeyen kavramlar; gerekirse tek yeniden yazım,
olmazsa bulgular aynen).

Denetim sayı ve cinsiyet/marka gibi kavramları yakalar; bulgularda olmayan bir *yorumu* her zaman yakalayamaz. Bunu istem
kuralları ve `--honesty` değerlendirmesi azaltır.

## Araçlar

| Araç | Ne yapar |
|---|---|
| `dataset_overview` | Hangi veri var, neler analiz edilemez |
| `category_preferences` | Bir demografik grubun bütçesinden diğerlerinden daha büyük/küçük pay ayırdığı kategoriler |
| `category_buyer_profile` | Bir kategoriyi hangi gruplar alıyor (alan hane oranı ve bütçe payı) |
| `compare_groups` | Harcama, sepet sayısı, sepet tutarı, kategori çeşidi, özel marka payı: gruplar arası |
| `segment_profile` | Tek bir grubun (örn. 65+, 125K+, çocuklu) profili |
| `demographic_drivers` | Hangi demografik özellik bir ölçütle en çok ilişkili |

## İstatistik yöntemleri

Her karşılaştırma **p-değeri, etki büyüklüğü ve bir karar** döndürür:

| Durum | Karar |
|---|---|
| p ≥ 0,05 | `no_significant_difference`: fark yok denir |
| p < 0,05 ama etki ihmal edilebilir | `significant_but_negligible_effect`: tercih diye sunulmaz |
| p < 0,05 ve etki küçük/orta/büyük | `meaningful_difference` |

Testler: Kruskal-Wallis (ε²) ve Mann-Whitney U (rank-biserial r) hane başına ölçütler için; ki-kare (yanlılığı düzeltilmiş Cohen w)
bir kategoriyi alan hane oranı için. "Hangi kategori öne çıkıyor?" sorusunda her hane için kategorinin **bütçe payı** hesaplanır
(az alışveriş yapan grup yapay olarak "az kategori alıyor" görünmesin diye), grup ile diğer haneler Mann-Whitney ile karşılaştırılır,
yüzlerce hücre için Benjamini-Hochberg yanlış keşif oranı düzeltmesi uygulanır; ayrıca fark en az 0,1 puan ve en az 1,3 kat olmalıdır.

## Test ve değerlendirme

```bash
cd backend
python -m unittest discover -s tests -t .
python -m sales_assistant.evaluation.runner --validate
python -m sales_assistant.evaluation.runner --plan
python -m sales_assistant.evaluation.runner --hard
python -m sales_assistant.evaluation.runner --honesty
```
