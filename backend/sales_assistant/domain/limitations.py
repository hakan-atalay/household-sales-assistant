from __future__ import annotations

DATASET_TITLE = "dunnhumby Complete Journey: bir ABD market zincirinin 801 hanesi, 2017"

LIMITATIONS: tuple[str, ...] = (
    "Demografi 801 hane için bilinir: yaş aralığı, yıllık gelir, hane büyüklüğü, hane tipi, çocuk sayısı. Cinsiyet bilgisi YOKTUR.",
    "Analiz birimi haneidir (kişi değil). Kayıtlar tek bir hanenin birçok alışverişidir; bu yüzden testler hane düzeyinde yapılır.",
    "Ürünler kategori düzeyindedir; marka bilgisi yalnızca ulusal marka / özel marka ayrımıdır.",
    "Tek bir yıl (2017) ve tek bir market zinciri: mevsim etkisi ve zincirin müşteri yapısı sonuçları etkiler; Türkiye'yi temsil etmez.",
    "Her demografik boyut ayrı test edilir; yaş, gelir ve çocuk durumu birbiriyle ilişkili olduğu için bir fark başka bir boyuttan gelebilir.",
    "Para birimi belirtilmemiştir (büyük olasılıkla ABD doları); tutarlar karşılaştırma içindir.",
)

EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "Yaşa göre ürün tercihi değişiyor mu?",
    "Çocuklu haneler hangi kategorilerde öne çıkıyor?",
    "Bebek maması en çok hangi gelir grubu alıyor?",
    "Gelir arttıkça özel marka tercihi değişiyor mu?",
    "Harcamayı en çok hangi demografik özellik etkiliyor?",
    "65 yaş üstü hanelerin profili nedir?",
    "Hane büyüklüğüne göre sepet tutarı değişiyor mu?",
    "Elinizde hangi veriler var?",
)
