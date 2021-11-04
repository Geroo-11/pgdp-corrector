import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import TextIO, Tuple, List


class PGDPDataset(Dataset):
    """ Esta clase permite cargar el libro que uno quiera dando la ID del mismo.
    
        Permite acceder a cada patrón del mismo (página)
    """

    def __init__(self, book_id : str, root : str = None, transform : callable = None):
        """
        Parámetros
        ----------
        book_id : str
            Identificador del libro a cargar como dataset

        root : str
            Directorio en donde buscar los archivos para construir el dataset

        transform : callable
            Transform que se va a aplicar sobre la muestra
        """

        self.book_id = book_id
        self.book_dir = root if root is not None else 'data/books'
        self.transform = transform
        self.samples = None
        self.targets = None
        self.letters = set()

        # Parseamos el texto y creamos una representacion matricial
        book_samples = f'projectID{book_id}_OCR.txt'
        book_targets = f'projectID{book_id}_P1_saved.txt'
        samples_path = self.book_dir + '/' + book_samples
        targets_path = self.book_dir + '/' + book_targets

        with open(samples_path, 'r') as f:
            self.samples = self._parse_pages(f)

        with open(targets_path, 'r') as f:
            self.targets = self._parse_pages(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        """ Cada item es una tupla (Sample, Target) """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_target = self.samples[idx], self.targets[idx]

        if self.transform:
            sample_target = self.transform(sample_target)

        return sample_target

    def _parse_pages(self, f : TextIO) -> List[np.ndarray]:
        # Este metodo toma un archivo de texto y devuelve una Lista de matrices con
        # las lineas de cada pagina
        reading_page = False
        lines = []
        pages = []
        for line in tqdm(f):
            # Agregamos las letras de la linea
            self.letters = self.letters | set(line)
            # Nos fijamos si la linea tiene contenido o no
            is_content = self._line_is_content(line)

            if reading_page and is_content:
                lines.append(line)
            elif line == '':
                if lines != []:
                    pages.append(np.array(lines))
                    lines = []
                break
            elif not is_content:
                if lines != []:
                    pages.append(np.array(lines))
                    lines = []
                reading_page = True
            else:
                pass
        return pages

    def _line_is_content(self, line : str) -> bool:
        # Este metodo toma una linea y devuelve True si es contenido
        # o Falso si es un separador
        return not line.startswith('-----File:')

data = PGDPDataset('616af0991b339')