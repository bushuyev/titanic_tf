import unittest
import numpy as np
import pandas as pd
from main import cabin_to_deck, add_missing_dummy_values


class TestDataPreparationMethods(unittest.TestCase):

    def test_cabin_to_deck_single(self):
        self.assertEqual(cabin_to_deck("C123"), "C")

    def test_cabin_to_deck_two(self):
        self.assertEqual(cabin_to_deck("B58 B60"), "B")

    def test_cabin_to_deck_missing(self):
        self.assertEqual(cabin_to_deck(""), "X")
        self.assertEqual(cabin_to_deck(None), "X")

    def test_add_missing_dummy_values(self):
        df_full = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
        df_to_fix = pd.DataFrame(np.array([[1, 2], [4, 5], [7, 8]]), columns=['a', 'b'])

        add_missing_dummy_values(df_full, df_to_fix, 0)

        self.assertSetEqual(set(df_to_fix.columns), set(df_full.columns))
        self.assertEqual(df_to_fix.loc[0, 'c'], 0)

if __name__ == '__main__':
    unittest.main()
