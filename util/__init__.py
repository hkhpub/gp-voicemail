import numpy as np


class GrowingMat(object):

    def __init__(self, shape, capacity, grow_factor=4):
        self.data = np.zeros(capacity)
        self.shape = shape
        self.capacity = capacity
        self.grow_factor = grow_factor

    def expand(self, cols=None, rows=None, block=None):
        if cols is not None and rows is not None:
            cols = np.atleast_2d(cols)
            rows = np.atleast_2d(rows)
            new_shape = (
                self.shape[0] + rows.shape[0], self.shape[1] + cols.shape[1])
            new_capacity = (
                self.capacity[0] * self.grow_factor if new_shape[
                    0] > self.capacity[0] else self.capacity[0],
                self.capacity[1] * self.grow_factor if new_shape[1] > self.capacity[1] else self.capacity[1])
            if new_capacity != self.capacity:
                # grow array
                newdata = np.zeros(new_capacity)
                newdata[:self.shape[0], :self.shape[1]] = self.data
                self.capacity = new_capacity
                self.data = newdata

            self.data[self.shape[0]:new_shape[0], :self.shape[1]] = rows
            self.data[:self.shape[0], self.shape[1]:new_shape[1]] = cols
            if block is not None:
                self.data[self.shape[0]:new_shape[0],
                          self.shape[1]:new_shape[1]] = block

            self.shape = new_shape
            #print "New shape", new_shape, self.shape, self.view.shape, #self.finalized.shape
        elif cols is not None:
            cols = np.atleast_2d(cols)
            new_shape = (self.shape[0], self.shape[1] + cols.shape[1])
            new_capacity = (self.capacity[0],
                            self.capacity[1] * self.grow_factor if new_shape[1] > self.capacity[1] else self.capacity[1])
            if new_capacity != self.capacity:
                # grow array
                newdata = np.zeros(new_capacity)
                newdata[:self.shape[0], :self.shape[1]] = self.data
                self.capacity = new_capacity
                self.data = newdata

            self.data[:self.shape[0], self.shape[1]:new_shape[1]] = cols
            self.shape = new_shape

        elif rows is not None:

            rows = np.atleast_2d(rows)
            new_shape = (self.shape[0] + rows.shape[0], self.shape[1])
            new_capacity = (
                self.capacity[0] * self.grow_factor if new_shape[
                    0] > self.capacity[0] else self.capacity[0],
                self.capacity[1])
            if new_capacity != self.capacity:
                # grow array
                newdata = np.zeros(new_capacity)
                newdata[:self.shape[0], :self.shape[1]] = self.data
                self.data = newdata
                self.capacity = new_capacity

            self.data[self.shape[0]:new_shape[0], :self.shape[1]] = rows
            self.shape = new_shape

    @property
    def view(self):
        return self.data[:self.shape[0], :self.shape[1]]

    @view.setter
    def view(self, d):
        self.data[:self.shape[0], :self.shape[1]] = d

    @property
    def finalized(self):
        data = self.view
        return np.reshape(data, newshape=self.shape)


class GrowingVector(object):

    def __init__(self, size, capacity=100, grow_factor=4):
        self.data = np.zeros(capacity)
        self.size = size
        self.capacity = capacity
        self.grow_factor = grow_factor

    def expand(self, rows):

        rows = np.atleast_1d(rows)
        new_size = self.size + rows.shape[0]
        new_capacity = self.capacity * \
            self.grow_factor if new_size > self.capacity else self.capacity
        if new_capacity != self.capacity:
            # grow array
            newdata = np.zeros(new_capacity)
            newdata[:self.size] = self.data
            self.capacity = new_capacity
            self.data = newdata

        self.data[self.size:new_size] = rows
        self.size = new_size

    @property
    def view(self):
        return self.data[:self.size]

    @view.setter
    def view(self, d):
        self.data[:self.size] = d

    @property
    def finalized(self):
        data = self.view
        return np.reshape(data, newshape=(self.size))
