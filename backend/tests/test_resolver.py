import unittest

from sales_assistant.data.resolver import EntityNotFound, EntityResolver
from tests.helpers import data_available, shared_repository


@unittest.skipUnless(data_available(), "household tables missing")
class ResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = EntityResolver(shared_repository())

    def test_categories_resolve_from_loose_text(self):
        for text, expected in {"imported wine": "IMPORTED WINE", "Diaper": "DIAPERS & DISPOSABLES", "soft drinks": "SOFT DRINKS"}.items():
            self.assertEqual(self.resolver.category(text), expected, text)

    def test_ambiguous_or_unknown_category_raises_with_suggestions(self):
        with self.assertRaises(EntityNotFound) as caught:
            self.resolver.category("wine")
        self.assertIn("IMPORTED WINE", caught.exception.suggestions)
        with self.assertRaises(EntityNotFound):
            self.resolver.category("xyzxyz")

    def test_segments(self):
        cases = {("kids", "çocuklu"): "Kids", ("kids", "cocuksuz"): "No kids", ("age", "65+"): "65+", ("age", "yaşlı"): "65+",
                 ("income", "yüksek"): "125K+", ("income", "125K"): "125K+", ("household_size", "5+"): "5+"}
        for (dimension, text), expected in cases.items():
            self.assertEqual(self.resolver.segment(dimension, text), expected, text)

    def test_unknown_segment_raises(self):
        with self.assertRaises(EntityNotFound):
            self.resolver.segment("age", "200")


if __name__ == "__main__":
    unittest.main()
