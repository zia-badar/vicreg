import pickle


class Result:

    def __init__(self, path):
        self.path = path

    def save(self):
        with open(self.path, 'wb') as file:
            pickle.dump(self, file)

    def load(self):
        with open(self.path, 'rb') as file:
            return pickle.load(file)