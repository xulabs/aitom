from MrcLoader import MrcLoader

if __name__ == '__main__':
    m = MrcLoader('test.mrc')
    m.read((0,0,0), (100, 100, 100), 0, True)
