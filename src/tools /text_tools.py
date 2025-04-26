import openai
from typing import Dict, Any
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import numpy as np

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

class TextTools:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
        except:
            self.sentiment_analyzer = None

    def summarize(self, text: str, max_length: int = 150) -> Dict[str, Any]:
        """
        Summarize the given text using OpenAI's API
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a text summarization expert. Create a concise summary of the following text."},
                    {"role": "user", "content": text}
                ],
                max_tokens=max_length,
                temperature=0.5
            )
            
            return {
                "success": True,
                "summary": response.choices[0].message["content"],
                "length": len(response.choices[0].message["content"].split())
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the given text
        """
        try:
            # If transformers pipeline is available, use it
            if self.sentiment_analyzer:
                result = self.sentiment_analyzer(text)[0]
                return {
                    "success": True,
                    "sentiment": result["label"],
                    "confidence": result["score"]
                }
            
            # Fallback to OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Analyze the sentiment of the following text. Respond with either POSITIVE, NEGATIVE, or NEUTRAL, followed by a confidence score between 0 and 1."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            
            result = response.choices[0].message["content"].strip().split("\n")[0]
            sentiment, confidence = result.split(" ")
            
            return {
                "success": True,
                "sentiment": sentiment,
                "confidence": float(confidence)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def translate(self, text: str, target_language: str) -> Dict[str, Any]:
        """
        Translate text to the target language
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Translate the following text to {target_language}. Provide only the translation."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            
            return {
                "success": True,
                "translation": response.choices[0].message["content"],
                "target_language": target_language
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def extract_keywords(self, text: str, max_keywords: int = 10) -> Dict[str, Any]:
        """
        Extract key terms and phrases from the text
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Extract up to {max_keywords} key terms or phrases from the following text. Provide them as a comma-separated list."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            
            keywords = [k.strip() for k in response.choices[0].message["content"].split(",")]
            
            return {
                "success": True,
                "keywords": keywords,
                "count": len(keywords)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def check_grammar(self, text: str) -> Dict[str, Any]:
        """
        Check grammar and suggest improvements
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Check the grammar of the following text. Provide corrections and explanations for any errors found."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            
            return {
                "success": True,
                "analysis": response.choices[0].message["content"],
                "original_text": text
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
