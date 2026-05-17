from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import compression_retriever

model = OllamaLLM(model="qwen2.5")

template = """
You are an expert crime lawyer and judge.

Use the provided legal cases and laws to:
1. Identify relevant legal principles
2. Apply them to the case
3. Give two clear judgment-style answer of found guilty and found innocent separately 
4. Also tell the likelyhood of guilty or not
5. At the end suggest some advise from a layers perspective 

Always reference relevant case laws when possible.

Relevant laws and cases:
{laws}

Case:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


while True:
    print("\n==============================")
    question = input("Enter case info (q to quit): ")

    if question.lower() == "q":
        break

    # 🔍 Retrieve relevant docs
    retrieved_docs = compression_retriever.invoke(question)

    laws_text = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in retrieved_docs
    ])

    # 🤖 Generate response
    result = chain.invoke({
        "laws": laws_text,
        "question": question
    })

    print("\n--- ⚖️ AI Response ---\n")
    print(result)
