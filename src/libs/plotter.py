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
    
    def plot_daily_consumption(self, period):
        daily_data = self.data_access.get_daily_consumption(period)
        if not daily_data:
            print("Não há dados para o período selecionado.")
            return
            
        df = pd.DataFrame(daily_data, columns=['Dia', 'Consumo (kWh)'])
        
        fig = px.line(df, x='Dia', y='Consumo (kWh)', 
                      title=f'Consumo Diário Total de Energia ({period.replace("_", " ").title()})',
                      markers=True)
        fig.update_layout(xaxis_title="Data", yaxis_title="Consumo Total (kWh)")
        fig.show()
    
    def plot_power_outliers(self, period):
        power_data = self.data_access.get_power_readings_by_device(period)
        if not power_data:
            print("Não há dados para o período selecionado.")
            return
            
        df = pd.DataFrame(power_data, columns=['Aparelho', 'Potência Ativa (kW)'])

        fig = px.box(df, x='Aparelho', y='Potência Ativa (kW)',
                     title=f'Distribuição de Potência e Anomalias por Aparelho ({period.replace("_", " ").title()})',
                     points="all") # 'all' mostra todos os pontos, destacando outliers
        fig.update_xaxes(tickangle=45) # Rotaciona os nomes dos aparelhos para melhor leitura
        fig.show()
    
    def plot_power_factor_analysis(self, device_id, device_name, period):
        pf_data = self.data_access.get_power_factor_analysis(device_id, period)
        if not pf_data:
            print(f"Não há dados para o aparelho {device_name} no período selecionado.")
            return
            
        df = pd.DataFrame(pf_data, columns=['Potência Ativa (kW)', 'Fator de Potência'])

        fig = px.scatter(df, x='Potência Ativa (kW)', y='Fator de Potência',
                         title=f'Análise de Eficiência: Fator de Potência vs. Consumo para {device_name}',
                         trendline="ols",  # Adiciona uma linha de tendência para ver a correlação
                         trendline_color_override="red")
        fig.update_yaxes(range=[0.5, 1.0]) # Fixa a escala do Fator de Potência
        fig.show()
