from abc import ABC

import pyodbc

from src.config.env import settings


class DataDB(ABC):
    def __init__(self):
        self.conn = pyodbc.connect(
            'DRIVER={PostgreSQL Unicode};'
            f'SERVER={settings.DB_POSTGRESQL_SERVER};'
            f'DATABASE={settings.DB_POSTGRESQL_DATA_DATABASE};'
            f'UID={settings.DB_POSTGRESQL_USER};'
            f'PWD={settings.DB_POSTGRESQL_PWD};'
            f'PORT={settings.DB_POSTGRESQL_PORT};'
        )
        self.cursor = self.conn.cursor()
        
class MemoryDB(ABC):
    def __init__(self):
        self.conn = pyodbc.connect(
            'DRIVER={PostgreSQL Unicode};'
            f'SERVER={settings.DB_POSTGRESQL_SERVER};'
            f'DATABASE={settings.DB_POSTGRESQL_MEMORY_DATABASE};'
            f'UID={settings.DB_POSTGRESQL_USER};'
            f'PWD={settings.DB_POSTGRESQL_PWD};'
            f'PORT={settings.DB_POSTGRESQL_PORT};'
        )
        self.cursor = self.conn.cursor()