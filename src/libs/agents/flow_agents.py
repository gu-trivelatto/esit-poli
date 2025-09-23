from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.libs.state import GraphStateType
from src.libs.agents.main_agents import AgentBase


class InputTranslator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are responsible of verifying if the USER_INPUT is in another language other
            than english, if so, translate the input. If the text is already in english
            simply output the same text. \n
            
            Your output must be a JSON object with two keys, 'language' and 'input', where
            'language' is the source language the user wrote and 'input' is the translated
            input. \n
        
            If the language is english, use ALWAYS the whole word to select it, never 'en',
            'eng', or any other variation. Use ALWAYS 'english'. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT : {user_input} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input"],
        )
    
    def execute(self) -> GraphStateType:
        user_input = self.state['user_input']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.ht_json_model | JsonOutputParser()

        llm_output = llm_chain.invoke({"user_input": user_input})
        translated_user_input = llm_output['input']
        source_language = llm_output['language']
        
        if self.debug:
            self.memory.save_debug("---TRANSLATE INPUT---")
            self.memory.save_debug(f'ORIGINAL INPUT: {user_input.rstrip()}')
            self.memory.save_debug(f'SOURCE LANGUAGE: {source_language}')
            if source_language.lower() != 'english':
                self.memory.save_debug(f'TRANSLATED INPUT:{translated_user_input.rstrip()}\n')
        
        self.state['num_steps'] = num_steps
        self.state['user_input'] = translated_user_input
        self.state['target_language'] = source_language
        
        return self.state
    
class ToolBypasser(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are part of the Energy System Insight Tool (ESIT), you are responsible for checking
            if the user trying to have a simple interaction with the tool instead of search for
            information or manipulate the model. \n

            Considering the USER_INPUT you should decide whether to route
            the user to the tool or simply bypass it to generate a simple answer to the user. \n
            
            The cases where you will bypass to the output are when USER_INPUT contains:
            - 'Hello!', 'Hi!', 'Hey!' or similars, without anything else;
            - 'How are you?' or similars, without anything else;
            - 'Who are you?' or similars;
            - 'What do you do?' or similars;
            - 'Thank you!', 'Thanks!', 'Thanks a lot!' or similars;
            
            You must output a JSON with a single key 'is_conversation' containing exclusivelly the 
            selected type, which can be either true or false (use full lowercase for the boolean). \n
            
            Always use double quotes in the JSON object. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT : {user_input} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input"],
        )
    
    def execute(self) -> GraphStateType:
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        user_input = self.state['user_input']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        llm_output = llm_chain.invoke({"user_input": user_input})
        is_conversation = llm_output['is_conversation']
        if type(is_conversation) == str:
            is_conversation = True if is_conversation == 'true' else False
        elif type(is_conversation) != bool:
            is_conversation = False
        
        if self.debug:
            self.memory.save_debug("---TYPE IDENTIFIER---")
            self.memory.save_debug(f'BYPASS TO OUTPUT: {is_conversation}\n')
        
        self.state['is_conversation'] = is_conversation
        self.state['num_steps'] = num_steps
        
        return self.state

class ContextAnalyzer(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert at analyzing the available CONTEXT and CHAT_HISTORY to decide if the available
            information is already enough to answer the question asked in USER_INPUT. \n

            It's important to know that you are the context analyzer for a branch of a tool, and this branch is
            responsible to gather general information to be used in the modeling context. If the USER_INPUT
            is related to modeling and uses information available at the data sources to apply modeling actions
            and all this information is available, then you should consider that the context is ready. Other
            tools will be responsible of using this data for modeling. \n
            
            If there is nothing related to modeling you can simply define it as ready when all information is
            gathered. \n
            
            Your output is a JSON object with a single key 'ready_to_answer', where you can use either true
            or false (always write it in lowercase). \n
            
            Always use double quotes in the JSON object. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            USER_INPUT: {user_input} \n
            CONTEXT : {context} \n
            CHAT_HISTORY: {history} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["user_input","context","history"]
        )
        
    def execute(self) -> GraphStateType:
        self.memory.save_chat_status('Processing')
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.json_model | JsonOutputParser()
        
        user_input = self.state['consolidated_input']
        context = self.state['context']
        history = self.state['history']
        num_steps = self.state['num_steps']
        num_steps += 1

        llm_output = llm_chain.invoke({"user_input": user_input, "context": context, "history": history})
        
        if self.debug:
            self.memory.save_debug("---TOOL SELECTION---")
            self.memory.save_debug(f'READY TO ANSWER: {llm_output["ready_to_answer"]}\n')
        
        self.state['next_query'] = llm_output
        self.state['num_steps'] = num_steps
        
        return self.state
    
class OutputTranslator(AgentBase):
    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are the final node of a tool, and you are responsible for translating the
            TOOL_OUTPUT from english to a given TARGET_LANGUAGE. \n
            
            Your output must be a JSON object with a single key, 'output', where you should
            put the translated output. \n
            
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            TOOL_OUTPUT : {tool_output} \n
            TARGET_LANGUAGE : {target_language} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["tool_output", "target_language"],
        )
    
    def execute(self) -> GraphStateType:
        final_answer = self.state['final_answer']
        target_language = self.state['target_language']
        num_steps = self.state['num_steps']
        num_steps += 1
        
        if self.debug:
            self.memory.save_debug("---TRANSLATE OUTPUT---")
            self.memory.save_debug(f'TARGET LANGUAGE: {target_language}\n')
        
        prompt = self.get_prompt_template()
        llm_chain = prompt | self.ht_json_model | JsonOutputParser()

        llm_output = llm_chain.invoke({"tool_output": final_answer, "target_language": target_language})
        
        self.state['num_steps'] = num_steps
        self.state['final_answer'] = llm_output['output']
        
        return self.state
