class Vocabulary(object):
    
    def __init__(self):
        self.length = 51
        self.index_to_word = {0: 'five', 1: 'r', 2: 'three', 3: 'j', 4: 'four', 5: 'green', 6: 'p', 7: 'q', 8: 'c', 9: 'one', 10: 'o', 11: 'seven', 12: 't', 13: 'x', 14: 'z', 15: 'g', 16: 'lay', 17: 'now', 18: 'please', 19: 'eight', 20: 'set', 21: 'by', 22: 'six', 23: 'nine', 24: 'v', 25: 'bin', 26: 'with', 27: 'two', 28: 'l', 29: 'i', 30: 'place', 31: 'u', 32: 'white', 33: 'y', 34: 'in', 35: 'at', 36: 'a', 37: 'b', 38: 'soon', 39: 'd', 40: 'm', 41: 'zero', 42: 'blue', 43: 'e', 44: 'n', 45: 'f', 46: 'again', 47: 'k', 48: 's', 49: 'h', 50: 'red'}
        self.word_to_index = {'five': 0, 'r': 1, 'three': 2, 'j': 3, 'four': 4, 'green': 5, 'p': 6, 'q': 7, 'c': 8, 'one': 9, 'o': 10, 'seven': 11, 't': 12, 'x': 13, 'z': 14, 'g': 15, 'lay': 16, 'now': 17, 'please': 18, 'eight': 19, 'set': 20, 'by': 21, 'six': 22, 'nine': 23, 'v': 24, 'bin': 25, 'with': 26, 'two': 27, 'l': 28, 'i': 29, 'place': 30, 'u': 31, 'white': 32, 'y': 33, 'in': 34, 'at': 35, 'a': 36, 'b': 37, 'soon': 38, 'd': 39, 'm': 40, 'zero': 41, 'blue': 42, 'e': 43, 'n': 44, 'f': 45, 'again': 46, 'k': 47, 's': 48, 'h': 49, 'red': 50}

    def init(self):
        pass

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        try:
            return self.index_to_word[key]
        except KeyError:
            return self.word_to_index[key]
