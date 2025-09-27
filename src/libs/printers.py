from abc import ABC, abstractmethod
from src.libs.state import GraphStateType
from src.libs.memory import Memory


class PrinterBase(ABC):
    def __init__(self, state: GraphStateType, debug):
        self.state = state
        self.debug = debug
        self.memory = Memory()
    
    @abstractmethod
    def execute(self) -> None:
        pass

class StatePrinter(PrinterBase):
    def execute(self) -> None:
        """print the state"""
        if self.debug:
            self.memory.save_debug("------------------STATE PRINTER------------------")
            self.memory.save_debug(f"Num Steps: {self.state['num_steps']} \n")
            self.memory.save_debug(f"Initial Query: {self.state['user_input']} \n" )
            self.memory.save_debug(f"Context: {self.state['context']} \n" )
        return

class FinalAnswerPrinter(PrinterBase):
    def execute(self) -> None:
        """prints final answer"""
        if self.debug:
            self.memory.save_debug("------------------FINAL ANSWER------------------")
            self.memory.save_debug(f"Final Answer: {self.state['final_answer']} \n")
            
        history = self.state['history']
        history.append({"role": "assistant", "content": self.state['final_answer']})
        
        self.memory.save_history(history)
        
        return