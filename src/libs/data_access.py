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

    def get_daily_consumption(self, period):
        now = datetime(2025, 9, 15)
        if period == "last_week":
            start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "last_month":
            start = (now - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "last_year":
             start = (now - timedelta(days=365)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError("Período inválido. Use 'last_week', 'last_month' ou 'last_year'.")
        
        end = now

        # Query para somar o consumo de todos os aparelhos, agrupado por dia.
        # DATE_TRUNC('day', m.timestamp) agrupa todos os timestamps para o início do dia.
        query = """
            SELECT 
                CAST(DATE_TRUNC('day', m.timestamp) AS DATE) as consumption_day, 
                SUM(m.active_power / 60.0) as total_kwh
            FROM measurements m
            WHERE m.timestamp BETWEEN ? AND ?
            GROUP BY consumption_day
            ORDER BY consumption_day;
        """
        self.cursor.execute(query, (start, end))
        rows = self.cursor.fetchall()
        
        # Converte explicitamente a lista de objetos Row em uma lista de tuplas
        results = [tuple(row) for row in rows]
        
        return results

    def get_power_readings_by_device(self, period):
        now = datetime(2025, 9, 15)
        if period == "yesterday":
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(hour=23, minute=59, second=59)
        elif period == "last_week":
            start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
        else:
            raise ValueError("Período inválido. Use 'yesterday' ou 'last_week'.")

        # Query para buscar a potência ativa de cada aparelho.
        query = """
            SELECT d.name, m.active_power
            FROM measurements m
            JOIN devices d ON m.device_id = d.device_id
            WHERE m.timestamp BETWEEN ? AND ?;
        """
        self.cursor.execute(query, (start, end))
        rows = self.cursor.fetchall()
        
        # Converte explicitamente a lista de objetos Row em uma lista de tuplas
        results = [tuple(row) for row in rows]
        
        return results

    def get_power_factor_analysis(self, device_id, period):
        now = datetime(2025, 9, 15)
        if period == "last_week":
            start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "last_month":
            start = (now - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            raise ValueError("Período inválido. Use 'last_week' ou 'last_month'.")
        
        end = now

        query = """
            SELECT m.active_power, m.power_factor
            FROM measurements m
            WHERE m.device_id = ? AND m.timestamp BETWEEN ? AND ?;
        """
        self.cursor.execute(query, (device_id, start, end))
        rows = self.cursor.fetchall()
        
        # Converte explicitamente a lista de objetos Row em uma lista de tuplas
        results = [tuple(row) for row in rows]
        
        return results

    def get_power_outliers(self, period):
        self.cursor.execute("""
            
        """, (period,))
        return self.cursor.fetchall()
        pass
