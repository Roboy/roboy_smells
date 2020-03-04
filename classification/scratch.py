import numpy as np

liste = ['a', 'a']

liste1 = ['b', 'b', 'b']

liste2 = ['c', 'c', 'c', 'c']
array = np.array(liste)


liste3 = [a for a in zip(liste, liste1, liste2)]

print(liste3)