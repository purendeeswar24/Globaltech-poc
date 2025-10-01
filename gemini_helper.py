import google.generativeai as genai
from typing import List, Optional
import os

# Initialize Gemini with your API key
def init_gemini(api_key: str):
    """Initialize the Gemini API with the provided API key."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro')

def generate_explanation(query: str, context: str, model) -> str:
    """Generate a clear and concise explanation using Gemini."""
    prompt = f"""You are a helpful biology tutor for NEET aspirants. 
    Explain the following concept in a simple, easy-to-understand way:
    
    Query: {query}
    
    Context from study material:
    {context}
    
    Provide a clear, step-by-step explanation with examples if possible.
    Keep it concise but comprehensive, suitable for a NEET aspirant.
    Use bullet points or numbered lists for better readability.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating explanation: {str(e)}")
        return "I couldn't generate an explanation at the moment. Here's the relevant information from the document:\n\n" + context

def enhance_answer_with_gemini(question: str, original_answer: str, model) -> str:
    """Enhance the original answer using Gemini for better clarity."""
    prompt = f"""You are a helpful biology tutor. Improve the following answer to make it more clear and engaging:
    
    Question: {question}
    
    Current Answer: {original_answer}
    
    Please:
    1. Keep it concise but comprehensive
    2. Use simple language
    3. Add bullet points or numbered lists if helpful
    4. Include examples or analogies if relevant
    5. Make it engaging for a NEET aspirant
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error enhancing answer: {str(e)}")
        return original_answer
