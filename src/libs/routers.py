from abc import ABC, abstractmethod

from src.libs.state import GraphStateType
from src.libs.memory import Memory


class BaseRouter(ABC):
    def __init__(self, state: GraphStateType, debug):
        self.state = state
        self.debug = debug
        self.memory = Memory()
    
    @abstractmethod
    def execute(self) -> str:
        pass
        
class BypassRouter(BaseRouter):
    def execute(self) -> str:
        """
        Route either to the tool or to the output.
        Args:
            state (dict): The current graph state
        Returns:
            str: Next node to call
        """
        is_conversation = self.state['is_conversation']
        
        message = "---BYPASS ROUTER---\nROUTE TO: "
        
        if is_conversation:
            message += "Skip to output\n"
            selection = "output"
        else:
            message += "Use tool\n"
            selection = "tool"

        if self.debug:
            self.memory.save_debug(message)
            
        return selection
    
class ToolRouter(BaseRouter):
    def execute(self) -> str:
        """
        Route to the necessary tool.
        Args:
            state (dict): The current graph state
        Returns:
            str: Next node to call
        """
        selection = self.state['selected_tool']
        
        message = "---TOOL ROUTER---\nROUTE TO: "
        
        if selection == 'web_search':
            message += "Web Search\n"
        elif selection == 'calculator':
            message += "Calculator\n"
        elif selection == "rag_search":
            message += "Rag Search\n"
        elif selection == "consult_data":
            message += "Consult Data\n"
            
        if self.debug:
            self.memory.save_debug(message)
            
        return selection

# TODO this router should be used also for the Actions router

class ContextRouter(BaseRouter):
    def execute(self) -> str:
        data_complete = self.state['is_data_complete']

        message = "---CONTEXT ROUTER---\nROUTE TO: "

        if data_complete:
            message += "Final answer generation\n"
            selection = "ready_to_answer"
        else:
            message += "Gather more context\n"
            selection = "need_context"
            
        if self.debug:
            self.memory.save_debug(message)
            
        return selection

class TranslationRouter(BaseRouter):
    def execute(self) -> str:
        target_language = self.state['target_language']
        
        message = "---TRANSLATOR ROUTER---\nROUTE TO: "
        
        if target_language.lower() == 'english':
            message += "print output"
            translate = False
        else:
            message += "translate output\n"
            translate = True
        
        return str(translate)