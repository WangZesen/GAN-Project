import os
import random
import numpy as np

class dataReader:
    def __init__(self, file_dir):
        self.pointer = 0
        self.file_dir = file_dir
        self.file_list = os.listdir(self.file_dir)
        random.shuffle(self.file_list)
        self.f_pointer = 0
        self.data = np.load(os.path.join(self.file_dir, self.file_list[0]))
        np.random.shuffle(self.data)

    def next_batch(self, batch_size = 10):
        if self.pointer + batch_size <= self.data.shape[0]:
            self.pointer += batch_size
            return self.data[self.pointer - batch_size: self.pointer]
        else:
            tmp_data = self.data[self.pointer: self.pointer + batch_size]
            self.f_pointer = (self.f_pointer + 1) % len(self.file_list)
            self.data = np.load(os.path.join(self.file_dir, self.file_list[self.f_pointer]))
            np.random.shuffle(self.data)
            batch_size -= tmp_data.shape[0]
            self.pointer = batch_size
            tmp_data = np.concatenate((tmp_data, self.data[0: batch_size]), axis = 0)
            return tmp_data
        
class labelDataReader:
    def __init__(self, file_dir):
        self.pointer = 0
        self.file_dir = file_dir
        file_list = os.listdir(self.file_dir)
        self.file_list = []
        for file in file_list:
            
            if 'image' in file:
                self.file_list.append(file)
                
        random.shuffle(self.file_list)
        self.f_pointer = 0
        self.data = np.load(os.path.join(self.file_dir, self.file_list[0]))
        self.label = np.load(os.path.join(self.file_dir, self.file_list[0].replace('image', 'label')))
        
        tmp = list(zip(self.data, self.label))
        random.shuffle(tmp)
        self.data = np.array(list(zip(*tmp))[0])
        self.label = np.array(list(zip(*tmp))[1])

    def next_batch(self, batch_size = 10):
        if self.pointer + batch_size <= self.data.shape[0]:
            self.pointer += batch_size
            return self.data[self.pointer - batch_size: self.pointer], self.label[self.pointer - batch_size: self.pointer]
        else:
            tmp_data = self.data[self.pointer: self.pointer + batch_size]
            tmp_label = self.label[self.pointer: self.pointer + batch_size]
            self.f_pointer = (self.f_pointer + 1) % len(self.file_list)
            self.data = np.load(os.path.join(self.file_dir, self.file_list[self.f_pointer]))
            self.label = np.load(os.path.join(self.file_dir, self.file_list[self.f_pointer].replace('image', 'label')))
            
            tmp = list(zip(self.data, self.label))
            random.shuffle(tmp)
            self.data = np.array(list(zip(*tmp))[0])
            self.label = np.array(list(zip(*tmp))[1])
            
            batch_size -= tmp_data.shape[0]
            self.pointer = batch_size
            tmp_data = np.concatenate((tmp_data, self.data[0: batch_size]), axis = 0)
            tmp_label = np.concatenate((tmp_label, self.label[0: batch_size]), axis = 0)
            return tmp_data, tmp_label

if __name__ == '__main__':
    data_reader = labelDataReader('../data/animate_label_subset/')

    for i in range(200):
        batch_x, batch_y = data_reader.next_batch(200)
        print (batch_x.shape, i)