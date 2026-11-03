import unittest
from decimal import Decimal

from scripts import sweep_medical_pced_gamma as sweep


class MedicalPCEDGammaSweepTest(unittest.TestCase):
    def test_gamma_parsing_and_output_names(self):
        self.assertEqual((Decimal("1"), Decimal("2.5"), Decimal("4")),
                         sweep.parse_gammas("1,2.5,4"))
        self.assertEqual("gamma_2p5", sweep.gamma_key(Decimal("2.5")))
        for bad in ("", "-1,2.5", "NaN", "1,1.0", "1,Infinity"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                sweep.parse_gammas(bad)

    def test_reference_gamma_is_the_only_reused_arm(self):
        self.assertEqual(Decimal("2.5"), sweep.REFERENCE_GAMMA)
        self.assertNotEqual(sweep.gamma_key(Decimal("1")),
                            sweep.gamma_key(Decimal("4")))


if __name__ == "__main__":
    unittest.main()
