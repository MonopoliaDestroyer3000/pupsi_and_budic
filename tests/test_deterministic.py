import tempfile
import unittest
from pathlib import Path

from harness.deterministic import solve


TAIL = " Work only inside the current directory. When you are done, write ONLY the final answer to the file answer.txt."


class DeterministicTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_math_mixed_operations_and_floor(self):
        story = ("A warehouse starts with 1752 boxes. Then the stock is split equally into 2 warehouses "
                 "and it keeps one share (rounded down). Then the stock is doubled by merging with a sister warehouse. "
                 "Then the stock is split equally into 3 warehouses and it keeps one share (rounded down). "
                 "Then the stock is doubled by merging with a sister warehouse. Then the stock is split equally "
                 "into 3 warehouses and it keeps one share (rounded down). How many boxes does it have at the end?")
        self.assertEqual(solve("math", story + TAIL, self.workspace)[0], "389")

    def test_math_deliveries(self):
        story = ("A warehouse starts with 1669 boxes. It ships 36 boxes per day for 3 days. "
                 "It receives 2 deliveries of 87 boxes each. How many boxes does it have at the end?")
        self.assertEqual(solve("math", story + TAIL, self.workspace)[0], "1735")

    def test_math_unknown_operation_abstains(self):
        story = ("A warehouse starts with 100 boxes. Ten boxes vanish overnight. "
                 "It receives 2 boxes. How many boxes does it have at the end?")
        self.assertIsNone(solve("math", story, self.workspace)[0])

    def test_math_zero_division_abstains(self):
        story = ("A warehouse starts with 100 boxes. Then the stock is split equally into 0 warehouses "
                 "and it keeps one share (rounded down). How many boxes does it have at the end?")
        self.assertIsNone(solve("math", story, self.workspace)[0])

    def sales(self):
        (self.workspace / "sales.csv").write_text(
            "order_id,region,product,month,units,price\n"
            "1,north,widget,7,10,2.5\n2,south,widget,8,20,4\n"
            "3,north,gadget,7,50,3\n4,north,widget,9,40,2\n", encoding="utf-8")

    def test_csv_sample_query(self):
        self.sales()
        instruction = "The file sales.csv has columns order_id,region,product,month,units,price (4 rows). How many units of product 'widget' were sold in months 7 through 8 inclusive?"
        self.assertEqual(solve("data_analysis", instruction, self.workspace)[0], "30")

    def test_csv_revenue_multiple_filters(self):
        self.sales()
        instruction = "The file sales.csv has sales data. What is the total revenue for product 'widget' in region 'north' in months 7 to 9 inclusive?"
        self.assertEqual(solve("data_analysis", instruction, self.workspace)[0], "105")

    def test_csv_order_count(self):
        self.sales()
        instruction = "The file sales.csv has sales data. How many orders for product 'widget' were placed in month 7?"
        self.assertEqual(solve("data_analysis", instruction, self.workspace)[0], "1")

    def test_csv_unknown_condition_abstains(self):
        self.sales()
        instruction = "The file sales.csv has sales data. How many units of product 'widget' were sold excluding canceled orders?"
        self.assertIsNone(solve("data_analysis", instruction, self.workspace)[0])

    def test_csv_missing_file_and_path_escape_abstain(self):
        for name in ("missing.csv", "../outside.csv"):
            instruction = f"The file {name} has sales data. How many units were sold?"
            self.assertIsNone(solve("data_analysis", instruction, self.workspace)[0])

    def test_csv_empty_average_abstains(self):
        self.sales()
        instruction = "The file sales.csv has sales data. What is the average price for product 'absent'?"
        self.assertIsNone(solve("data_analysis", instruction, self.workspace)[0])

    def test_csv_nonterminating_average_abstains(self):
        (self.workspace / "sales.csv").write_text("price\n0\n0\n2\n", encoding="utf-8")
        instruction = "The file sales.csv has sales data. What is the average price?"
        self.assertIsNone(solve("data_analysis", instruction, self.workspace)[0])

    def test_csv_conflicting_conditions_abstain(self):
        self.sales()
        instruction = "The file sales.csv has sales data. How many units of product 'widget' and product 'gadget' were sold?"
        self.assertIsNone(solve("data_analysis", instruction, self.workspace)[0])

    def kb(self):
        root = self.workspace / "kb"
        root.mkdir()
        (root / "person_a.md").write_text("# Irina Smith\n\nBorn: 1983\nEmployer: Fjord Data\nMentor: Sergei Garcia\n", encoding="utf-8")
        (root / "person_b.md").write_text("# Sergei Garcia\n\nBorn: 1985\nEmployer: Fjord Data\n", encoding="utf-8")
        (root / "company.md").write_text("# Fjord Data\n\nHeadquarters: Osaka\nFounded: 1965\n", encoding="utf-8")
        return root

    def test_search_direct_and_relation_hops(self):
        self.kb()
        for question, answer in [
            ("In which year was Irina Smith born?", "1983"),
            ("In which city is the headquarters of Sergei Garcia's employer?", "Osaka"),
            ("In which city is the headquarters of Irina Smith's mentor's employer?", "Osaka"),
            ("In which year was Irina Smith's mentor born?", "1985"),
            ("What is Irina Smith's employer?", "Fjord Data"),
        ]:
            with self.subTest(question=question):
                self.assertEqual(solve("search", "The directory kb/ contains markdown files. " + question + TAIL, self.workspace)[0], answer)

    def test_search_missing_and_conflicting_evidence_abstain(self):
        root = self.kb()
        self.assertIsNone(solve("search", "In which year was Unknown Person born?", self.workspace)[0])
        (root / "duplicate.md").write_text("# Irina Smith\nBorn: 1999\n", encoding="utf-8")
        self.assertIsNone(solve("search", "In which year was Irina Smith born?", self.workspace)[0])

    def test_unknown_type_and_multiple_questions_abstain(self):
        self.assertIsNone(solve("coding", "Anything", self.workspace)[0])
        self.assertIsNone(solve("search", "In which year was Irina Smith born? What is her employer?", self.workspace)[0])


if __name__ == "__main__":
    unittest.main()
