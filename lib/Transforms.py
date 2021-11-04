import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, List

class Flatten(object):
    """ Esta transformación 'achata' cada página. Las convierte en una cadena unidimensional
        de caracteres """
    def __init__(self):
        pass

    def __call__(self, item : Tuple[np.ndarray, np.ndarray]) -> Tuple[str, str]:
        sample, target = item[0], item[1]
        new_sample = ''.join(list(sample))
        new_target = ''.join(list(target))
        return new_sample, new_target

class OneHot(object):
    """ Esta transformación convierte cada página en un vector one-hot con la dimensionalidad
        especificada. Tiene un metodo para invertir."""
    def __init__(self, letters : List[str]):
        """ Parámetros
        ----------
        letters : List[str]
            Lista de letras contenidas en tanto los samples como los targets. Estas se obtienen
            automaticamente al crear un PGDPDataset.
        """
        # Se agrega el caracter 'NUL' para poder hacer padding si es necesario
        letters.append('\0')
        self.encoder = OneHotEncoder()
        self.flatten = Flatten()
        self.encoder.fit(np.array(letters)[:,None])
        self.size = len(self.encoder.categories_)

    def __call__(self, item : Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        sample, target = self.flatten(item)
        sample, target = np.array(list(sample))[:, None], np.array(list(target))[:, None]
        return self.encoder.transform(list(sample)), self.encoder.transform(list(target))