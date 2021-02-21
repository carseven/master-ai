import os
import matplotlib.pyplot as plt
import numpy as np

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
ejemplo = plt.imread("/Users/carseven/dev/master-ai/percepcion/img/fruta.jpg")

# %%
arr = np.array([[[1, 2, 3],
                 [3, 4, 5]],
                [[1, 2, 3],
                 [3, 4, 5]],
                [[1, 2, 3],
                 [3, 4, 5]],
                [[1, 2, 3],
                 [3, 4, 5]]])
print(arr.shape)

# arr = np.asarray(ejemplo, dtype='float')
# background = (1, 1, 1)
# background = np.ravel(background).astype(arr.dtype)

# background = background[np.newaxis, ...]
# alpha = arr[..., -1, np.newaxis]
# channels = arr[np.newaxis, ..., :-1]

# out = np.squeeze(np.clip((1 - alpha) * background + alpha * channels,
#                          a_min=0, a_max=1),
#                  axis=0)

# plt.imshow(out, cmap='gray')
# plt.show()
