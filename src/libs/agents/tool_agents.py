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
            2. no_op
            
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
        else:
            str_result = 'The requested data operation can not be performed, stop the execution and inform the user'
        
        if self.debug:
            self.memory.save_debug(f'RESULT: {str_result}\n')
            
        self.state['context'] = context + [str_result]
        self.state['num_steps'] = num_steps
        
        return self.state