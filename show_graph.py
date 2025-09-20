from src.chat_llm import GraphBuilder

def main():
    print("Welcome to the Energy System Insight Tool (ESIT)")
    GraphBuilder('CESM/Data/Techmap', [], False).display_graph()

if __name__ == '__main__':
    main()