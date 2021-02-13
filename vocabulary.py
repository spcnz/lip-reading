class Vocabulary(object):
    
    def __init__(self):
        self.word_to_index_map = {}

        self.index_to_word_map = {}

        self.occurrences_map = {}
        
        self.length = 0

        self.total = 0


    def extend(self, words):
        word_set = set(words)
        
        self.total += len(words)

        for index, word in enumerate(word_set):
            if (word not in self.word_to_index_map):
                self.word_to_index_map[word] = index
                self.index_to_word_map[index] = word
                self.length += 1


    def __len__(self):
        print('daaaa', self.length)
        return self.length

    def __getitem__(self, key):
        try:
            return self.index_to_word_map[key]
        except KeyError:
            return self.word_to_index_map[key]
