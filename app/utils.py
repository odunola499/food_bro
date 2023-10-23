from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA





class Func:
    def __init__(self):
        self.parser_llm = OpenAI(temperature = 0.9, model = 'gpt-3.5-turbo-instruct', max_tokens = -1)
        compressor = LLMChainExtractor.from_llm(self.parser_llm)
        embedding_function = HuggingFaceEmbeddings(model_name='LargeEmbedder')
        small_context_index = Chroma(persist_directory="chroma_db", embedding_function=embedding_function, collection_name = 'foodie_small').as_retriever()
        large_context_index = Chroma(persist_directory="chroma_db", embedding_function=embedding_function, collection_name = 'foodie').as_retriever(search_kwargs = {'k':2})
        self.compression_retriever_small = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=small_context_index)
        self.compression_retriever_large = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=large_context_index)
        template = """You are a chef that helps design and recommend great food and recipe, healthy or not. YOu are very knowledgeable in food related discussions. Use the following pieces of context to answer the food related request at the end.
        Take note of the following points that must all be strictly adhered toas you respond
        1. If you don't know the answer, just say that you don't know, don't try to make up an answer or use answers from your own thoughts.
        2.  Be very explanatory in your answer as the user but ensure all you say is retrieved from the context given to you.
        3.  If you recommend or you create a food dish or food menu from the context given always provide the ingredient and directions on how to the dish, each of them in seperate bullet points, 
        4. Be very enthusiastic in your replies and be with lots of vibe ad energy!, make jokes even
        THese three points are extremely important instructions. Do not deviate!

        {context}
        Request: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        self.qa_chain_small = RetrievalQA.from_chain_type(
        self.parser_llm,
        retriever=small_context_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
        self.qa_chain_large = RetrievalQA.from_chain_type(
        self.parser_llm,
        retriever=large_context_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
        self.qa_chain_with_compression_small = RetrievalQA.from_chain_type(
        self.parser_llm,
        retriever=self.compression_retriever_small,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
        self.qa_chain_with_compression_large = RetrievalQA.from_chain_type(
        self.parser_llm,
        retriever=self.compression_retriever_large,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
        
        
    def RAG_short_content(self, question, use_compression = False):
        if use_compression:
            result = self.qa_chain_with_compression_small({"query":question})
            return result['result'], result['source_documents']
        result = self.qa_chain_small({"query":question})
        return result['result'], result['source_documents']
    def RAG_large_content(self, question, use_compression = False):
        if use_compression:
            result = self.qa_chain_with_compression_large({"query":question})
            return result['result'], result['source_documents']
        result = self.qa_chain_large({"query":question})
        return result['result'], result['source_documents']
        
      

