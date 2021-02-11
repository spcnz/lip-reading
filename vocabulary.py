class Vocabulary(object):
    
    def __init__(self, words):
        self.word_to_index_map = {}

        self.index_to_word_map = {}

        self.occurrences_map = {}

        word_set = set(words)
        
        self.total = len(words)
        self.length = len(word_set)

        for index, word in enumerate(word_set):
            self.word_to_index_map[word] = index
            self.index_to_word_map[index] = word
            self.occurrences_map[word] = 0

        for word in words:
            self.occurrences_map[word] += 1

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        try:
            return self.index_to_word_map[key]
        except KeyError:
            return self.word_to_index_map[key]


    def occurrences(self, word=None):
        if word:
            return self.occurrences_map[word]
        else:
            return self.occurrences_map
