# LangChain Summarization Techniques

This document explains the summarization techniques provided by LangChain—**Stuff**, **MapReduce**, and **Refine**—with detailed examples and use cases.

---

## 1. Stuff Summarization

### **Overview:**
- Combines all input documents into a single string and passes them to the LLM for summarization.
- **Best for small to medium-sized documents** within the model's context window.

### **Pros:**
- Simple and fast.
- Lower cost due to a single API call.

### **Cons:**
- Limited by the LLM's token limit.
- Not suitable for large documents.

### **Example:**
```python
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI

loader = TextLoader("example.txt")
documents = loader.load()

llm = ChatOpenAI(temperature=0, model="gpt-4")

prompt = PromptTemplate(input_variables=["text"], template="Summarize the following text:\n{text}\nSummary:")
chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

summary = chain.run(documents)
print(summary)
```

---

## 2. MapReduce Summarization

### **Overview:**
- Splits documents into chunks, summarizes each chunk (Map step), and merges summaries (Reduce step).
- **Best for large documents** exceeding the model's token limit.

### **Pros:**
- Scalable for large datasets.
- Suitable for parallel processing.

### **Cons:**
- May produce fragmented summaries.
- Higher costs due to multiple LLM calls.

### **Example:**
```python
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI

loader = TextLoader("large_example.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

llm = ChatOpenAI(temperature=0, model="gpt-4")

map_prompt = PromptTemplate(template="Summarize this chunk: {text}", input_variables=["text"])
combine_prompt = PromptTemplate(template="Combine these summaries into a final summary: {text}", input_variables=["text"])

chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt)

summary = chain.run(docs)
print(summary)
```

---

## 3. Refine Summarization

### **Overview:**
- Iteratively improves the summary by processing chunks one by one.
- **Best for maintaining context in long and complex documents.**

### **Pros:**
- Preserves context and coherence.
- Produces high-quality summaries for lengthy documents.

### **Cons:**
- Slowest method due to iterative refinement.
- Higher cost because of multiple refinement steps.

### **Example:**
```python
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI

loader = TextLoader("large_example.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

llm = ChatOpenAI(temperature=0, model="gpt-4")

prompt = PromptTemplate(template="Summarize this text:\n{text}", input_variables=["text"])
refine_prompt = PromptTemplate(
    template="We have the existing summary: {existing_summary}\nRefine it with this additional context: {text}",
    input_variables=["existing_summary", "text"]
)

chain = load_summarize_chain(llm, chain_type="refine", question_prompt=prompt, refine_prompt=refine_prompt)

summary = chain.run(docs)
print(summary)
```

---

## Comparison Table: Stuff vs MapReduce vs Refine

| Feature                 | Stuff                        | MapReduce                    | Refine                        |
|-------------------------|------------------------------|------------------------------|-------------------------------|
| **Best For**             | Small to medium documents    | Large documents              | Long, complex documents       |
| **Workflow**             | Single-pass processing       | Multi-pass (Map + Reduce)    | Iterative refinement          |
| **Context Handling**     | Limited by context size      | Splits context into chunks   | Maintains incremental context |
| **Speed**                | Fast                         | Slower than Stuff            | Slowest due to iteration       |
| **Cost**                 | Low API cost (single call)   | Medium (multiple calls)      | High (multiple iterative calls)|
| **Output Coherence**     | Good for short summaries     | May be fragmented             | Highly coherent               |

---

## Key Takeaways
- Use **Stuff** for small documents where simplicity is preferred.
- Opt for **MapReduce** for large datasets with distributed summarization.
- Choose **Refine** for complex texts where coherence and incremental refinement are required.

---

## References
- [LangChain Documentation](https://python.langchain.com/docs/)
- [OpenAI GPT-4 Documentation](https://platform.openai.com/docs/)

