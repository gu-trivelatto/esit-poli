from abc import ABC
from src.config.db import DataDB
from src.libs.data_access import DataAccess
import plotly.express as px
import plotly.io as pio
import pandas as pd


class Plotter(ABC):
    def __init__(self):
        db = DataDB()
        cursor = db.cursor
        self.data_access = DataAccess(cursor)
        pio.renderers.default = "browser"
    
    def plot_consumption_distribution(self, period):
        dist = self.data_access.get_consumption_distribution(period)
        labels = [f"{tipo.capitalize()}" for tipo in dist.keys()]
        values = [round(valor, 1) for valor in dist.values()]
        df = pd.DataFrame({'Tipo': labels, 'Consumo (kWh)': values})
        
        fig = px.pie(df, values='Consumo (kWh)', names='Tipo',
                     title='Distribuição de Consumo por Tipo de Aparelho')
        fig.show()
    
    def plot_power_outliers(self, period):
        pass
