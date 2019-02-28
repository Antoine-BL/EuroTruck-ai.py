from scripts.balance_data import balance_data
from scripts.mirror_images import mirror_images
from scripts.shuffle_data import shuffle_data
from scripts.split_data import split_data


def main():
    mirror = input('Mirror? (Y/N):').upper() == 'Y'
    balance = input('Balance? (Y/N):').upper() == 'Y'
    shuffle = input('Shuffle? (Y/N):').upper() == 'Y'
    split = input('Split? (Y/N):').upper() == 'Y'

    if mirror:
        print('mirroring data')
        mirror_images()

    if balance:
        print('balancing data')
        balance_data()

    if shuffle:
        print('shuffling data')
        shuffle_data()

    if split:
        print('splitting data')
        split_data()


if __name__ == "__main__":
    main()
