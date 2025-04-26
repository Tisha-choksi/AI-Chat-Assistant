import openai
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SearchTools:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.document_vectors = None
        self.documents = []

    def add_documents(self, documents: List[str]) -> Dict[str, Any]:
        """
        Add documents to the search index
        """
        try:
            self.documents = documents
            self.document_vectors = self.vectorizer.fit_transform(documents)
            
            return {
                "success": True,
                "document_count": len(documents),
                "vocabulary_size": len(self.vectorizer.vocabulary_)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def similarity_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Find most similar documents to the query
        """
        if not self.documents:
            return {
                "success": False,
                "error": "No documents in the index"
            }

        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            top_indices = similarities.argsort()[-top_k:][::-1]
            results = [
                {
                    "document": self.documents[idx],
                    "similarity_score": float(similarities[idx])
                }
                for idx in top_indices
            ]

            return {
                "success": True,
                "results": results
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def question_answering(self, question: str, context: str) -> Dict[str, Any]:
        """
        Answer questions based on provided context
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Answer the question based on the provided context. If the answer cannot be found in the context, say 'I cannot answer this question based on the provided context.'"},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ],
                temperature=0.3
            )
            
            return {
                "success": True,
                "answer": response.choices[0].message["content"],
                "context_used": context[:200] + "..." if len(context) > 200 else context
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def generate_related_questions(self, text: str, num_questions: int = 5) -> Dict[str, Any]:
        """
        Generate related questions based on the provided text
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Generate {num_questions} relevant questions based on the following text. Make the questions diverse and insightful."},
                    {"role": "user", "content": text}
                ],
                temperature=0.7
            )
            
            questions = [q.strip() for q in response.choices[0].message["content"].split("\n") if q.strip()]
            
            return {
                "success": True,
                "questions": questions,
                "count": len(questions)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def extract_topics(self, text: str, num_topics: int = 5) -> Dict[str, Any]:
        """
        Extract main topics from the text
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Extract {num_topics} main topics from the following text. For each topic, provide a brief description."},
                    {"role": "user", "content": text}
                ],
                temperature=0.5
            )
            
            topics = [t.strip() for t in response.choices[0].message["content"].split("\n") if t.strip()]
            
            return {
                "success": True,
                "topics": topics,
                "count": len(topics)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def semantic_search(self, query: str, documents: List[str], top_k: int = 3) -> Dict[str, Any]:
        """
        Perform semantic search using OpenAI embeddings
        """
        try:
            # Get query embedding
            query_response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_embedding = query_response['data'][0]['embedding']

            # Get document embeddings
            doc_response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=documents
            )
            doc_embeddings = [item['embedding'] for item in doc_response['data']]

            # Calculate similarities
            similarities = [
                cosine_similarity(
                    np.array(query_embedding).reshape(1, -1),
                    np.array(doc_embedding).reshape(1, -1)
                )[0][0]
                for doc_embedding in doc_embeddings
            ]

            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = [
                {
                    "document": documents[idx],
                    "similarity_score": float(similarities[idx])
                }
                for idx in top_indices
            ]

            return {
                "success": True,
                "results": results
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
