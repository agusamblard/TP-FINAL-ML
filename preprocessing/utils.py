import pandas as pd
import re
from difflib import get_close_matches


def quitar_tildes(texto):

    """
    Elimina tildes y caracteres especiales de un texto.
    """
    if pd.isna(texto):
        return ''
    texto = str(texto).strip().lower()

    # Reemplazo manual de tildes y otros caracteres especiales
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
    }
    texto = ''.join(reemplazos.get(c, c) for c in texto)

    return texto



def normalizar(texto, eliminar_espacios=True):
    if pd.isna(texto):
        return ''
    texto = str(texto).strip().lower()

    # Reemplazo de tildes y diéresis
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
        'ñ': 'n'
    }
    texto = ''.join(reemplazos.get(c, c) for c in texto)

    # Símbolos raros a eliminar
    simbolos_a_eliminar = ['-', '.', ',', '"', "'", '“', '”', '’', '`',
                           '(', ')', '[', ']', '{', '}', ':', ';', '!', '?', '#', '@', '°', 'º', 'ª', '/', '\\', '|']

    if eliminar_espacios:
        simbolos_a_eliminar.append(' ')

    for simbolo in simbolos_a_eliminar:
        texto = texto.replace(simbolo, '')

    # Eliminar cualquier carácter no alfanumérico restante
    if eliminar_espacios:
        texto = re.sub(r'[^a-z0-9]', '', texto)
    else:
        texto = re.sub(r'[^a-z0-9 ]', '', texto)

    return texto


def tokens_similares(s1, s2, cutoff=0.8):
    """
    Devuelve True si s1 y s2 comparten al menos un token similar
    (coincidencia exacta o fuzzy).
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        return False
    t1 = normalizar(s1, eliminar_espacios=False).split()
    t2 = normalizar(s2, eliminar_espacios=False).split()
    for token in t1:
        if token in t2 or get_close_matches(token, t2, n=1, cutoff=cutoff):
            return True
    return False
