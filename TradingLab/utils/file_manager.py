# Gestione file

"""
Gestione dei file per il progetto TradingLab.
Fornisce utilità per operazioni sui file come lettura/scrittura, compressione, e gestione CSV.
"""
import os
import csv
import json
import pickle
import gzip
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO, TextIO, Iterator

# Import dei logger
from .logger import app_logger
from .exceptions import TradingLabException, handle_exception

# Import delle configurazioni
from ..config import CACHE_DIR, TEMP_DIR


class FileError(TradingLabException):
    """Errore relativo alle operazioni sui file."""
    pass


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Assicura che una directory esista, creandola se necessario.
    
    Args:
        path: Percorso della directory
        
    Returns:
        Oggetto Path della directory
        
    Raises:
        FileError: Se non può creare la directory
    """
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            path_obj.mkdir(parents=True, exist_ok=True)
            app_logger.debug(f"Creata directory: {path_obj}")
        return path_obj
    except Exception as e:
        raise FileError(f"Impossibile creare la directory {path}", 
                       details={"exception": str(e)})


def safe_file_write(path: Union[str, Path], content: Union[str, bytes], 
                   mode: str = 'w', encoding: Optional[str] = 'utf-8',
                   use_temp: bool = True) -> None:
    """
    Scrive in modo sicuro il contenuto in un file usando un file temporaneo.
    
    Args:
        path: Percorso del file
        content: Contenuto da scrivere
        mode: Modalità di apertura ('w' per testo, 'wb' per binario)
        encoding: Codifica del file (ignorato in modalità binaria)
        use_temp: Se usare un file temporaneo per scrittura atomica
        
    Raises:
        FileError: Se non può scrivere nel file
    """
    path_obj = Path(path)
    
    # Assicurati che la directory esista
    ensure_directory(path_obj.parent)
    
    try:
        if use_temp:
            # Crea un file temporaneo nella stessa directory
            with tempfile.NamedTemporaryFile(
                delete=False, 
                dir=path_obj.parent,
                mode=mode, 
                encoding=encoding if 'b' not in mode else None
            ) as temp_file:
                temp_path = temp_file.name
                temp_file.write(content)
            
            # Rinomina atomicamente il file temporaneo
            shutil.move(temp_path, path_obj)
        else:
            # Scrittura diretta
            with open(path_obj, mode=mode, encoding=encoding if 'b' not in mode else None) as f:
                f.write(content)
        
        app_logger.debug(f"File scritto con successo: {path_obj}")
    
    except Exception as e:
        raise FileError(f"Impossibile scrivere nel file {path}", 
                       details={"exception": str(e)})


def safe_file_read(path: Union[str, Path], mode: str = 'r', 
                  encoding: Optional[str] = 'utf-8',
                  default: Any = None) -> Any:
    """
    Legge in modo sicuro il contenuto da un file.
    
    Args:
        path: Percorso del file
        mode: Modalità di apertura ('r' per testo, 'rb' per binario)
        encoding: Codifica del file (ignorato in modalità binaria)
        default: Valore da restituire se il file non esiste
        
    Returns:
        Contenuto del file o default se il file non esiste
        
    Raises:
        FileError: Se non può leggere il file
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        app_logger.debug(f"File non trovato: {path_obj}")
        return default
    
    try:
        with open(path_obj, mode=mode, encoding=encoding if 'b' not in mode else None) as f:
            content = f.read()
        
        app_logger.debug(f"File letto con successo: {path_obj}")
        return content
    
    except Exception as e:
        raise FileError(f"Impossibile leggere il file {path}", 
                       details={"exception": str(e)})


def save_json(path: Union[str, Path], data: Any, indent: int = 4, 
             ensure_ascii: bool = False) -> None:
    """
    Salva dati in formato JSON.
    
    Args:
        path: Percorso del file
        data: Dati da salvare
        indent: Indentazione JSON
        ensure_ascii: Se limitare i caratteri a ASCII
        
    Raises:
        FileError: Se non può salvare i dati
    """
    try:
        content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
        safe_file_write(path, content, 'w', 'utf-8')
    except Exception as e:
        raise FileError(f"Impossibile salvare i dati JSON in {path}", 
                       details={"exception": str(e)})


def load_json(path: Union[str, Path], default: Any = None) -> Any:
    """
    Carica dati dal formato JSON.
    
    Args:
        path: Percorso del file
        default: Valore da restituire se il file non esiste
        
    Returns:
        Dati caricati o default se il file non esiste
        
    Raises:
        FileError: Se non può caricare i dati
    """
    try:
        content = safe_file_read(path, 'r', 'utf-8', default=None)
        if content is None:
            return default
        
        return json.loads(content)
    except Exception as e:
        raise FileError(f"Impossibile caricare i dati JSON da {path}", 
                       details={"exception": str(e)})


def save_pickle(path: Union[str, Path], data: Any, compress: bool = False) -> None:
    """
    Salva dati in formato pickle.
    
    Args:
        path: Percorso del file
        data: Dati da salvare
        compress: Se comprimere i dati
        
    Raises:
        FileError: Se non può salvare i dati
    """
    try:
        if compress:
            with gzip.open(path, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        
        app_logger.debug(f"Dati pickle salvati in: {path}")
    except Exception as e:
        raise FileError(f"Impossibile salvare i dati pickle in {path}", 
                       details={"exception": str(e)})


def load_pickle(path: Union[str, Path], default: Any = None, 
               compressed: bool = False) -> Any:
    """
    Carica dati dal formato pickle.
    
    Args:
        path: Percorso del file
        default: Valore da restituire se il file non esiste
        compressed: Se i dati sono compressi
        
    Returns:
        Dati caricati o default se il file non esiste
        
    Raises:
        FileError: Se non può caricare i dati
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        app_logger.debug(f"File pickle non trovato: {path_obj}")
        return default
    
    try:
        if compressed:
            with gzip.open(path_obj, 'rb') as f:
                return pickle.load(f)
        else:
            with open(path_obj, 'rb') as f:
                return pickle.load(f)
    
    except Exception as e:
        raise FileError(f"Impossibile caricare i dati pickle da {path}", 
                       details={"exception": str(e)})


def save_csv(path: Union[str, Path], data: List[Dict[str, Any]], 
            fieldnames: Optional[List[str]] = None) -> None:
    """
    Salva dati in formato CSV.
    
    Args:
        path: Percorso del file
        data: Lista di dizionari da salvare
        fieldnames: Elenco dei campi da includere (se None, usa le chiavi del primo dizionario)
        
    Raises:
        FileError: Se non può salvare i dati
    """
    if not data:
        app_logger.warning(f"Nessun dato da salvare in CSV: {path}")
        return
    
    try:
        # Assicurati che la directory esista
        ensure_directory(Path(path).parent)
        
        # Ottieni i nomi dei campi
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        with open(path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        app_logger.debug(f"Dati CSV salvati in: {path}")
    
    except Exception as e:
        raise FileError(f"Impossibile salvare i dati CSV in {path}", 
                       details={"exception": str(e)})


def load_csv(path: Union[str, Path], default: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Carica dati dal formato CSV.
    
    Args:
        path: Percorso del file
        default: Valore da restituire se il file non esiste
        
    Returns:
        Lista di dizionari o default se il file non esiste
        
    Raises:
        FileError: Se non può caricare i dati
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        app_logger.debug(f"File CSV non trovato: {path_obj}")
        return default if default is not None else []
    
    try:
        with open(path_obj, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)
    
    except Exception as e:
        raise FileError(f"Impossibile caricare i dati CSV da {path}", 
                       details={"exception": str(e)})


def get_temp_path(prefix: str = "tradinglab_", suffix: str = "") -> Path:
    """
    Crea un percorso temporaneo per file temporanei.
    
    Args:
        prefix: Prefisso per il nome del file
        suffix: Suffisso per il nome del file (es. estensione)
        
    Returns:
        Percorso del file temporaneo
    """
    # Assicurati che la directory temporanea esista
    ensure_directory(TEMP_DIR)
    
    # Crea un nome file temporaneo
    temp_file = tempfile.NamedTemporaryFile(delete=False, prefix=prefix, suffix=suffix, dir=TEMP_DIR)
    temp_path = Path(temp_file.name)
    temp_file.close()
    
    return temp_path


def get_cache_path(key: str, subdir: Optional[str] = None) -> Path:
    """
    Genera un percorso per file di cache.
    
    Args:
        key: Chiave univoca per il file cache
        subdir: Sottodirectory opzionale
        
    Returns:
        Percorso per il file cache
    """
    import hashlib
    
    # Crea un hash della chiave per usarlo come nome file
    hash_obj = hashlib.md5(key.encode())
    filename = hash_obj.hexdigest()
    
    # Costruisci il percorso completo
    if subdir:
        cache_path = CACHE_DIR / subdir / filename
        ensure_directory(CACHE_DIR / subdir)
    else:
        cache_path = CACHE_DIR / filename
        ensure_directory(CACHE_DIR)
    
    return cache_path


def clean_temp_files(days_old: int = 1) -> int:
    """
    Pulisce i file temporanei più vecchi di un certo periodo.
    
    Args:
        days_old: Età minima in giorni per eliminare i file
        
    Returns:
        Numero di file eliminati
    """
    import time
    
    now = time.time()
    count = 0
    
    # Calcola il limite di tempo
    seconds_old = days_old * 86400
    
    try:
        for path in TEMP_DIR.glob("**/*"):
            if path.is_file():
                mtime = path.stat().st_mtime
                if now - mtime > seconds_old:
                    path.unlink()
                    count += 1
        
        app_logger.info(f"Puliti {count} file temporanei vecchi di {days_old} giorni")
        return count
    
    except Exception as e:
        app_logger.error(f"Errore durante la pulizia dei file temporanei: {e}")
        return count