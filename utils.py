import numpy as np

# def seq_padding(X, padding=0):
#     L = [len(x) for x in X]
#     ML = max(L)
#     return np.array([
#         np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
#     ])

# set the progress bar

class ShowProcess():

	# 

    i = 0 
    max_steps = 0 
    max_arrow = 50 
    infoDone = 'Done!'

    def __init__(self, max_steps, infoDone = 'Done!'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0

# if __name__=='__main__':
#     max_steps = 100

#     process_bar = ShowProcess(max_steps, 'Done!')

#     for i in range(max_steps):
#         process_bar.show_process()
#         time.sleep(0.01)

class data_generator:
    
    def __init__(self, data, para=None, label, batch_size=64):
        self.data = data
        if para:
            self.para = para
        self.label = label
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        def __len__(self):
            return self.steps
        def __iter__(self):
            while True:
                idxs = range(len(self.data))
                np.random.shuffle(idxs)
                X1, X2, Y = [], [], []
                for i in idxs:
                    X1.append(data[i])
                    if para:
                        X2.append(para[i])
                    Y.append(labels[i])
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        if para:
                            yield [X1, X2, Y]
                            X1, X2, Y = [], [], []
                        else:
                            yield [X1, Y]
                            X1, Y = [], []
                
    