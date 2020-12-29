# %%
import numpy as np
import time

# %% Clock list vs numpy array
rango = 10000000

list1 = range(rango)
list2 = range(rango)
array1 = np.array(range(rango))
array2 = np.array(range(rango))

start1 = time.time()
result = [x - y for x, y in zip(list1, list2)]
end1 = time.time()

start2 = time.time()
result = array1 - array2
end2 = time.time()

print('List:', end1 - start1)
print('Numpy arrays:', end2 - start2)

# %%
lista = [[1, 2, 3, 4, 5, 7, 8, 9, 10],
         [1, 2, 3, 4, 5, 7, 8, 9, 10]]
a = np.array(lista, dtype='float32')

print(a)
