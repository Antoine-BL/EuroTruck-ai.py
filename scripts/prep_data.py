from scripts.balance_data import balance_data
from scripts.mirror_images import mirror_images
from scripts.split_data import split_data
from scripts.visualise_datasets import plot_all


def main():
    # print('mirroring data')
    # mirror_images()
    print('balancing data')
    balance_data()
    print('splitting data')
    split_data()


if __name__ == "__main__":
    main()
