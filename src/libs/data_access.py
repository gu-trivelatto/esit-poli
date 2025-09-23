from abc import ABC

from pyodbc import Cursor
from datetime import datetime, timedelta


class DataAccess(ABC):
    def __init__(self, cursor: Cursor):
        self.cursor = cursor

    def get_consumption_distribution(self, period):
        # Define o intervalo de datas conforme o período
        now = datetime(2025, 9, 15)
        if period == "yesterday":
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(hour=23, minute=59, second=59)
        elif period == "last_week":
            start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        elif period == "last_month":
            start = (now - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        else:
            raise ValueError("Período inválido")

        query = """
            SELECT d.type, SUM(m.active_power) as total_active_power
            FROM devices d
            JOIN measurements m ON d.device_id = m.device_id
            WHERE m.timestamp BETWEEN ? AND ?
            GROUP BY d.type
        """
        self.cursor.execute(query, (start, end))
        results = self.cursor.fetchall()

        # Retorna como dicionário: {tipo: consumo_total}
        return {row.type: row.total_active_power / 60.0 for row in results}

    def get_power_outliers(self, period):
        self.cursor.execute("""
            
        """, (period,))
        return self.cursor.fetchall()
        pass
