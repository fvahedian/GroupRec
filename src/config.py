class Config(object):
    def __init__(self):
        self.embedding_size = 64
        self.layers = [int(i * self.embedding_size) for i in [2, 4, 2, 1]]
        self.epoch = 20
        self.batch_size = 128
        self.lr = [0.01, 0.001, 0.0005]
        self.drop_ratio = 0.2
        self.ks = [5, 10, 15, 20]

    def __str__(self):
        return ''.join(
                ['embedding_size: ', str(self.embedding_size), '\n',
                    'layers: ', str(self.layers), '\n', 
                    'batch_size: ', str(self.batch_size), '\n', 
                    'lr: ', str(self.lr), '\n', 
                    'drop_ratio: ', str(self.drop_ratio), '\n']
                )
