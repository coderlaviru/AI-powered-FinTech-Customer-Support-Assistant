# AI-powered-FinTech-Customer-Support-Assistant

Build an AI-powered FinTech Customer Support Assistant that can answer user questions using a provided financial knowledge base.

## 1. Problem Statement

Build an AI-powered FinTech Customer Support Assistant that can answer user questions using a provided financial knowledge base.

The system should use Retrieval-Augmented Generation (RAG) to retrieve relevant information from the provided documents before generating an answer.

The assistant should provide accurate, grounded responses and should clearly indicate when the required information is not available in the knowledge base.

## 2. Knowledge Base

A financial question-answering/document dataset is provided as part of the assignment.

Candidates should build their RAG pipeline using the provided data.

You may choose your own:

- Chunking strategy
- Embedding model
- Vector database
- Retrieval strategy
- LLM
- Prompting approach

**Note:** Your technical choices should be explained in the final video.

## 3. Core Requirements

Build a pipeline to:

- Load the provided financial documents
- Clean and preprocess the data
- Split documents into appropriate chunks
- Generate embeddings
- Store and retrieve relevant information

The system should implement:

```text
User Query → Retrieval → Context → LLM → Grounded Answer
```

The retrieved context should be relevant to the user's question.

### Customer Support Interface

Build a simple interface where users can:

- Ask financial questions
- View generated answers
- View the sources/references used to generate the answer
- Ask follow-up questions where appropriate

### Grounded Responses

The assistant should:

- Answer using the provided knowledge base
- Avoid making unsupported claims
- Clearly communicate when information cannot be found
- Provide relevant document/source references wherever possible

**Example:**

```text
Answer: The applicable foreclosure charge is 3% of the outstanding principal.
Source: Personal Loan Policy — Section 4.2
```

## 4. Evaluation & AI/ML Requirements

Candidates should demonstrate an understanding of:

- Data preprocessing
- Chunking strategies
- Embeddings
- Vector search
- Retrieval strategies
- Prompt engineering
- LLM selection
- Hallucination mitigation
- RAG evaluation

Candidates should define and evaluate suitable metrics such as:

- Retrieval quality
- Answer correctness
- Groundedness
- Citation/source accuracy

A small evaluation set should be created or used to demonstrate the effectiveness of the implemented RAG pipeline.

## 5. System Design

The solution should include an architecture similar to:

```text
Documents → Preprocessing → Chunking → Embeddings → Vector Store → Retriever → LLM → Response
```

Candidates should make appropriate technology choices and justify them.

Possible technologies include:

- Python
- FastAPI / Flask / Django
- LangChain / LlamaIndex
- FAISS / Chroma / Qdrant / pgvector / Pinecone
- OpenAI / Gemini / other LLM providers
- React / Next.js for the interface

**Note:** These are examples only; candidates may choose suitable alternatives.

## 6. Bonus Features

The following are optional:

- Conversation memory
- Hybrid search
- Reranking
- Query rewriting
- Streaming responses
- Confidence/relevance scoring
- Multiple document types
- Evaluation dashboard
- Automated evaluation
- Caching
- Dockerization
- Unit/integration tests
- Observability/logging
- Cost and latency optimization

## 7. Submission Requirements

Submit a GitHub repository containing:

- Complete source code
- README
- Setup instructions
- Environment variable documentation
- RAG architecture
- Model/vector database details
- API documentation, if applicable

Submit a publicly accessible URL of the working application.

If an API is used, provide the relevant API endpoint/documentation.

### 5-Minute Explanation Video

Maximum duration: 5 minutes

The video should explain:

- Problem understanding
- Overall system architecture
- Data preprocessing approach
- Chunking strategy
- Embedding/model selection
- Retrieval approach
- Prompt/LLM strategy
- Evaluation methodology
- Key trade-offs
- Challenges encountered
- Short demonstration of the application
- What you would improve with more time

## 8. Evaluation Criteria

| Area | Weightage |
|---|---:|
| RAG Quality & Answer Accuracy | 25% |
| Retrieval & Evaluation Strategy | 20% |
| AI/ML Understanding | 15% |
| System Design & Architecture | 15% |
| Code Quality | 10% |
| UI/UX & Usability | 5% |
| Documentation & Explanation | 10% |

## 9. Final Note

The goal is not to build a chatbot wrapper around an LLM. We are looking for a solution that demonstrates an understanding of RAG architecture, information retrieval, AI/ML concepts, evaluation, hallucination mitigation, and practical AI engineering.
