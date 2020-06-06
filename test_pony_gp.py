import sys
import unittest
import pony_gp


class TestPonyGP(unittest.TestCase):
    def test_one_main(self) -> None:
        """Test the `main` method of `pony_gp` with the `configs.ini` file

        """
        args = ["pony_gp.py", "--config=configs.ini"]
        sys.argv = args
        pony_gp.main()
        self.assertTrue(True)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestPonyGP("test_one_main"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
