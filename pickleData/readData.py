import os
import pickle


PICKLE_FILE = 'pickle.dat'


def main():
    # append data to the pickle file
    add_to_pickle(PICKLE_FILE, 123)
    add_to_pickle(PICKLE_FILE, 'Hello')
    add_to_pickle(PICKLE_FILE, None)
    add_to_pickle(PICKLE_FILE, b'World')
    add_to_pickle(PICKLE_FILE, 456.789)
    # load & show all stored objects
    for item in read_from_pickle(PICKLE_FILE):
        print(repr(item))
    os.remove(PICKLE_FILE)

def add_to_pickle(path, item):
    with open(path, 'wb') as file:
        pickle.dump(item, file, pickle.HIGHEST_PROTOCOL)


def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass


if __name__ == '__main__':
    main()


#import pickle
#data = pickle.load(open("data/save.p", "rb" ) )
#print(data[3])