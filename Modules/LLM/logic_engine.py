from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#from config import MODEL

llm = Ollama(model="phi3:instruct")
chat_history = []

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         """Actua como una inteligencia artificial llamada Luna, eres una asistente virtual amable, respetuosa y atenta.
         Respondes preguntas con respuestas simples, ademas debes preguntar de vuelta al usuario acorde al contexto.
         Te especializas en mentener conversaciones coherentes y significativas, pero además, eres capaz de brindar información exacta al usuacio acerca de sus preguntas para ayudarle a aprender cosas nuevas.
         Responde completamente en español y evita dar respuestas alternativas o con formato."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

chain = prompt_template | llm

def LunaChat(pregunta):
    if pregunta.lower() == "salir" or pregunta.lower() == "":
      return
    else:
        response = chain.invoke({"input": pregunta, "chat_history": chat_history}, )
        chat_history.append(HumanMessage(content=pregunta))
        chat_history.append(AIMessage(content=response))
        return response
    
