from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.libs.agents.main_agents import AgentBase
from src.libs.state import GraphStateType

    
class Calculator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an specialist at identifying the correct operation and operands to build
            a JSON object with the correct equation to be solved, you will get the information of
            what equation to generate from the given QUERY. \n
            
            You must output a JSON object with a single key 'equation', that must be in a format
            solvable with Python, since we'll solve this using the eval() function. You don't need
            to import anything or prepare the environment in any way, the environment is ready,
            you just need to build the equation in a format recognizable by python, using python
            mathematical functions if necessary. \n
            
            NEVER USE IMPORT STATEMENTS, YOU MUST ASSUME THAT ALL NECESSARY LIBRARIES ARE ALREADY
            IMPORTED, JUST USE THEM. \n
            
            If you need information that you don't know to build the equation you may find that
            in the CONTEXT, so check it for more details on the data. \n
            
            If you need intermediate results, you can calculate only them and leave the final
            result for later, you're part of a iterative tool that may be called as many times
            as needed to find the final result. \n
            
            Always use double quotes in the JSON object. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY: {query} \n
            CONTEXT: {context}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query","context"],
        )
        
    def execute(self) -> GraphStateType:
        self.memory.save_chat_status('Calculating result')
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
    
        query = self.state['user_input']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        llm_output = llm_chain.invoke({"query": query, "context": context})
        equation = llm_output['equation']
        
        if self.debug:
            self.memory.save_debug("---CALCULATOR TOOL---")
            self.memory.save_debug(f'EQUATION: {equation}')
        
        result = eval(equation)
        
        str_result = f'{equation} = {result}'
        
        if self.debug:
            self.memory.save_debug(f'RESULT: {str_result}\n')
            
        self.state['context'] = context + [str_result]
        self.state['num_steps'] = num_steps
        
        return self.state
    
class DataAgent(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an specialist at identifying the correct operation to be performed over the
            system's data, you will get a list of operations that you can perform, along with
            the necessary parameters for each operation. You will also get the QUERY with the
            user request, and the CONTEXT with the information you have until now. \n
            
            Given all this, your job is to select the correct operation to be performed,
            provide the necessary parameters for it, and decide if the user wants to see
            the plotted data based on the QUERY and CONTEXT. \n
            
            The available operations are: \n
            1. get_consumption_distribution(period) - period can be "yesterday", "last_week" or "last_month"
            2. get_daily_consumption(period) - period can be "last_week", "last_month" or "last_year"
            3. get_power_readings_by_device(period) - period can be "yesterday" or "last_week"
            4. get_power_factor_analysis(device_id, period) - device_id is an integer, period can be "last_week" or "last_month"
            5. no_op \n
            
            Here is the device_id list if needed: \n
            - Computador 1 => ID: 1
            - Monitor 1 => ID: 2
            - Computador 2 => ID: 3
            - Monitor 2 => ID: 4
            - Computador 3 => ID: 5
            - Monitor 3 => ID: 6
            - Computador 4 => ID: 7
            - Monitor 4 => ID: 8
            - Computador 5 => ID: 9
            - Monitor 5 => ID: 10
            - Computador 6 => ID: 11
            - Monitor 6 => ID: 12
            - Computador 7 => ID: 13
            - Monitor 7 => ID: 14
            - Computador 8 => ID: 15
            - Monitor 8 => ID: 16
            - Computador 9 => ID: 17
            - Monitor 9 => ID: 18
            - Computador 10 => ID: 19
            - Monitor 10 => ID: 20
            - Computador 11 => ID: 21
            - Monitor 11 => ID: 22
            - Computador 12 => ID: 23
            - Monitor 12 => ID: 24
            - Roteador => ID: 25
            - Projetor => ID: 26
            - Ar-condicionado 1 => ID: 27
            - Ar-condicionado 2 => ID: 28 \n

            You must output a JSON object with three keys, 'operation', that is the operation name
            as written in the provided list, 'parameters', which is a list with the parameters
            in the order they appear in the operation definition, and 'plot', which is a boolean
            indicating if the user wants to see the plotted data. \n
            
            If you don't know the answer, you must choose the operation 'no_op' with no parameters,
            and plot as false. \n

            Always use double quotes in the JSON object. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            QUERY: {query} \n
            CONTEXT: {context}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["query","context"],
        )
        
    def execute(self) -> GraphStateType:
        self.memory.save_chat_status('Getting data')
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
    
        query = self.state['user_input']
        context = self.state['context']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        llm_output = llm_chain.invoke({"query": query, "context": context})
        operation = llm_output['operation']
        parameters = llm_output['parameters']
        plot = llm_output['plot']

        if self.debug:
            self.memory.save_debug("---DATA TOOL---")
            self.memory.save_debug(f'OPERATION: {operation}')
            self.memory.save_debug(f'PARAMETERS: {parameters}')
            self.memory.save_debug(f'PLOT: {plot}')

        if operation == 'get_consumption_distribution':
            period = parameters[0]
            dist = self.plotter.data_access.get_consumption_distribution(period)
            labels = [f"{tipo.capitalize()}" for tipo in dist.keys()]
            values = [round(valor, 1) for valor in dist.values()]
            str_result = f'Consumo por tipo de aparelho em {period}: ' + ', '.join([f'{labels[i]}: {values[i]} kWh' for i in range(len(labels))])
            
            if plot:
                self.plotter.plot_consumption_distribution(period)
                str_result += ' [PLOT SHOWN]'
        elif operation == 'get_daily_consumption':
            period = parameters[0]
            daily_data = self.plotter.data_access.get_daily_consumption(period)
            if not daily_data:
                str_result = f'Não há dados para o período {period}.'
            else:
                str_result = f'Consumo diário total em {period}: ' + ', '.join([f'{row[0]}: {round(row[1], 2)} kWh' for row in daily_data])
                
                if plot:
                    self.plotter.plot_daily_consumption(period)
                    str_result += ' [PLOT SHOWN]'
        elif operation == 'get_power_readings_by_device':
            period = parameters[0]
            
            self.plotter.plot_power_outliers(period)
            str_result = 'The data is too extense for textual description, the plot was displayed to the user. [PLOT SHOWN]'
        elif operation == 'get_power_factor_analysis':
            device_id = parameters[0]
            period = parameters[1]
            pf_data = self.plotter.data_access.get_power_factor_analysis(device_id, period)
            if not pf_data:
                str_result = f'Não há dados para o aparelho ID {device_id} no período {period}.'
            else:
                str_result = f'Análise de fator de potência para o aparelho ID {device_id} em {period}: ' + ', '.join([f'Potência: {round(row[0], 2)} kW, Fator de Potência: {round(row[1], 2)}' for row in pf_data])
                
                if plot:
                    device_name = next((row[0] for row in pf_data if row[0] == device_id), f'Aparelho {device_id}')
                    self.plotter.plot_power_factor_analysis(device_id, device_name, period)
                    str_result += ' [PLOT SHOWN]'
        else:
            str_result = 'The requested data operation can not be performed, stop the execution and inform the user'
        
        if self.debug:
            self.memory.save_debug(f'RESULT: {str_result}\n')
            
        self.state['context'] = context + [str_result]
        self.state['num_steps'] = num_steps
        
        return self.state