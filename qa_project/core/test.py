from utils.faq.qa_system import QASystem

qa = QASystem()
result = qa.query("¿Cuál es el tema principal?")
print(result['result'])