import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.black_karasinski import BlackKarasinskiModel


def main():
    bk_model = BlackKarasinskiModel(a=0.1, sigma=0.1)
    bond_price = bk_model.zero_coupon_bond_price(T=5, r=0.05)


if __name__ == "__main__":
    main()
