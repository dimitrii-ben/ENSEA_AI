# import
from langchain import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
#import sentence_transformers
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import sys
import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")
#Parsing the pdf document and split it into chunks over lapping of 200 char.
# load the document
loader = UnstructuredPDFLoader(sys.argv[1])
documents = loader.load()
print("Starting script...")

print("Parsing the current document...")
# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
# create the open-source embedding function using HuggingFaceEmbeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_function = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# load it into Chroma (we do not persist for our presentation but we could've use persist)
# If persist is used and we have multiple document we are working with it is important to specify some more data about the 
# document in the metadata
print("Moving all the embeddings to the vector store. Please wait...")
db2 = Chroma.from_documents(docs, embedding_function)
print("All documents have been loaded\n\n")
#Defining our current LLM : currently using openAI chatgpt3.5-turbo
llm = OpenAI(model_name="gpt-3.5-turbo-0125",,temperature=0.2)
chain = load_qa_chain(llm,chain_type="stuff")

EOF = False
if (len(docs) <=5):
    k = len(docs)
while(not(EOF)):
    
    current_input = input("Query:")
    if(current_input != 'exit'):
        docs =db2.similarity_search(current_input,k=k)
        print(chain.run(input_documents=docs,question=current_input))
        
    else:
        EOF=True
        print("==========================\nTerminating the script\n==========================\n")
    
    
    


