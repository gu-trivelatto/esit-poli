from abc import ABC, abstractmethod
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.libs.plotter import Plotter
from src.libs.state import GraphStateType
from src.libs.memory import Memory


# TODO standardize variable names used from the state
# TODO standardize the way the agents interact with the state

class AgentBase(ABC):
    def __init__(self, llm_models, es_models, state: GraphStateType, app, debug):
        self.chat_model = llm_models['chat_model']
        self.json_model = llm_models['json_model']
        self.ht_model = llm_models['ht_model']
        self.ht_json_model = llm_models['ht_json_model']
        self.state = state
        self.debug = debug
        self.app = app
        self.selected_value = None
        self.base_model = es_models['base_model']
        self.mod_model = es_models['mod_model']
        self.memory = Memory()
        self.plotter = Plotter()
        
    def confirm_selection(self, selected_value):
        self.selected_value = selected_value

    @abstractmethod
    def get_prompt_template(self) -> PromptTemplate:
        pass

    def execute(self) -> GraphStateType:
        return self.state

# TODO the outputs should also indicate if the model was runned etc...

class OutputGenerator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are part of the Energy System Insight Tool (ESIT), a tool tailored
            to use LLM agents to help users to consult data and understand the
            details of a university laboratory energy consumption. You are the
            last agent of the tool, responsible for summing up the results generated
            by other agents in the output to be given to the user. If the user
            asks about you, present yourself as the tool itself, not as an agent. \n
            
            Given the USER_INPUT and a CONTEXT, generate an answer for the query
            asked by the user. You should make use of the provided information
            to answer the user in the best possible way. If you think the answer
            does not answer the user completely, ask the user for the necessary
            information if possible. \n
            
            Since you are part of a energy usage analyzing tool you will probably
            receive queries asking for historic data, comparisons, plots, etc.
            Never make up data or indicate plots for yourself. Other nodes
            already consulted the data and showed plots to the user, you should
            just forward this information to the user. For listing information
            about parameters you should show a topic list to the user if possible.
            Your main goal in data request scenarios is to give the user a
            comprehensive summary of the gathered data to help understand what
            is currently happening the in lab. You must show the user the gathered
            data as well. \n
            
            CHAT_HISTORY can also be used to gather context and information about
            past messages exchanged between you and the user. \n
            
            Also, you can look at the ACTION_HISTORY to understand which actions were
            executed regarding the system. \n
            
            If and only if the context you use to answer the user provides you a
            data source you should display the provided sources as follows:
            Source:
            - <url/document>
            - <url/document>
            - And so on for how many sources are needed... \n
            
            NEVER output an empty source, if there is no provided source DON'T
            ADD THE SOURCE LIST. THE PRESENCE OF THE SOURCE LIST IS OPTIONAL,
            IF THERE ARE NO SOURCES IN THE DATA YOU USED TO GENERATE THE ANSWER, THEN
            DON'T CREATE THE SOURCE LIST. \n            
            
            It's important never to cite the variables you receive, answer the most
            naturally as possible trying to make it as it was a simple conversation. \n
            
            The current datetime is {datetime}. \n

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            CONTEXT: {context} \n
            ACTION_HISTORY: {action_history} \n
            CHAT_HISTORY: {history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input","context","action_history","history"],
        )
        
    def execute(self) -> GraphStateType:
        self.memory.save_chat_status('Generating output')
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.ht_model | StrOutputParser()
        
        ## Get the state
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_input = self.state['user_input']
        context = self.state['context']
        action_history = self.state['action_history']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"datetime": date, "user_input": user_input, "context": context, "action_history": action_history, "history": history})
        
        if self.debug:
            self.memory.save_debug("---GENERATE OUTPUT---")
            self.memory.save_debug(f'GENERATED OUTPUT:\n{llm_output}\n')
        
        if '\nSource:\n- None' in llm_output:
            llm_output = llm_output.replace('\nSource:\n- None','')
            
        self.state['num_steps'] = num_steps
        self.state['final_answer'] = llm_output
        
        return self.state
    