import unittest

from sales_assistant.llm import grounding

EVIDENCE = '{"groups":[{"group":"North","n":2060,"median":95.0,"share_pct":17.293}],"p_value":0.0,"period":"2017-11"}'
FACTS = "112101 98199 27 73"


class GroundingTests(unittest.TestCase):
    def numbers(self, answer, question="Kuzey'de fiyat nedir?"):
        return grounding.check(answer, EVIDENCE, question, FACTS).numbers

    def test_numbers_from_results_pass_in_turkish_formats(self):
        self.assertEqual(self.numbers("Ortanca fiyat 95 (n=2.060), pay %17,3."), [])
        self.assertEqual(self.numbers("Toplam 112.101 kalem, Kasım 2017 (2017-11)."), [])

    def test_invented_numbers_are_flagged(self):
        self.assertEqual(self.numbers("Fiyat %77 yüksek, 1500 oldu."), [77.0, 1500.0])

    def test_list_markers_are_not_numbers(self):
        self.assertEqual(self.numbers("1. Birinci\n2. İkinci"), [])

    def test_unsupported_terms_follow_the_question(self):
        report = grounding.check("Kadınlar pahalı alıyor.", EVIDENCE, "Kuzey'de fiyat nedir?", FACTS)
        self.assertEqual(report.terms, ["kadınlar"])
        report = grounding.check("Bu veride cinsiyet bilgisi yok.", EVIDENCE, "Erkekler ne alıyor?", FACTS)
        self.assertEqual(report.terms, [])

    def test_english_dataset_word_is_flagged(self):
        self.assertEqual(grounding.unsupported_terms("Dataset'te yaş var.", "yaş?", "- 801 hane"), ["dataset"])

    def test_age_is_a_supported_concept(self):
        self.assertEqual(grounding.check("Yaşlı haneler daha az harcıyor.", EVIDENCE, "Kuzey'de fiyat nedir?", FACTS).terms, [])

    def test_terms_that_the_findings_themselves_contain_are_allowed(self):
        report = grounding.check("fashion_male_clothing (erkek giyim) öne çıkıyor", "- fashion_male_clothing (erkek giyim): payı %5.0", "Hangi kategori?", FACTS)
        self.assertTrue(report.ok)
        report = grounding.check("Erkekler daha çok alıyor.", "- Bölgeler arasında fark YOK", "Hangi bölge?", FACTS)
        self.assertEqual(report.terms, ["erkekler"])


if __name__ == "__main__":
    unittest.main()