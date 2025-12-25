from typing import Dict, List
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class AgentService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-3.5-turbo"  # Can be configured via environment

    async def process_documentation_query(self, query: str, context: List[Dict] = None, language: str = "en") -> str:
        """Process a documentation-related query using OpenAI"""
        if context is None:
            context = []

        # Build the context for the AI
        context_str = ""
        if context:
            if language.lower() in ["ur", "urdu"]:
                context_str = "متعلقہ دستاویزات کا متن:\n"
                for doc in context:
                    content_preview = doc.get('content_preview', '')[:200]
                    # If the content is in English but user wants Urdu, we may need to translate
                    context_str += f"- {doc.get('title', 'Unknown')}: {content_preview}...\n"
            else:
                context_str = "Relevant documentation context:\n"
                for doc in context:
                    context_str += f"- {doc.get('title', 'Unknown')}: {doc.get('content_preview', '')[:200]}...\n"

        # Create the prompt based on the language
        if language.lower() in ["ur", "urdu"]:
            prompt = f"""
            آپ دستاویزات کے لیے ایک AI اسسٹنٹ ہیں۔ فراہم کردہ دستاویزات کے متن کی بنیاد پر صارف کے سوال کا جواب دیں۔

            {context_str}

            صارف کا سوال: {query}

            دستاویزات کی بنیاد پر مددگار اور درست جواب فراہم کریں۔ اگر دستاویزات میں ضروری معلومات نہیں ہے تو واضح طور پر کہیں۔
            """
            system_message = "آپ ایک مددگار دستاویزات اسسٹنٹ ہیں۔ فراہم کردہ دستاویزات کے متن کی بنیاد پر درست جوابات فراہم کریں۔ مختصر لیکن جامع رہیں۔ جواب اردو میں دیں۔"
        else:
            prompt = f"""
            You are an AI assistant for documentation. Answer the user's question based on the provided documentation context.

            {context_str}

            User question: {query}

            Provide a helpful, accurate answer based on the documentation. If the documentation doesn't contain the information needed, say so clearly.
            """
            system_message = "You are a helpful documentation assistant. Provide accurate answers based on the documentation context provided. Be concise but thorough."

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=700,
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()

            # If the user requested Urdu but the response is in English, attempt to translate
            if language.lower() in ["ur", "urdu"]:
                # In a real implementation, we'd ensure the response is in Urdu
                # For now, we'll return the result as is, assuming the model followed the instructions
                pass

            return result
        except Exception as e:
            logger.error(f"Error processing query with OpenAI: {str(e)}")
            if language.lower() in ["ur", "urdu"]:
                return "معاف کریں، لیکن آپ کی استفسار کو حل کرتے وقت مجھے ایک خرابی کا سامنا کرنا پڑا۔ براہ کرم بعد میں دوبارہ کوشش کریں۔"
            else:
                return "I'm sorry, but I encountered an error while processing your query. Please try again later."

    async def generate_explanation(self, text: str, language: str = "en") -> str:
        """Generate an explanation for selected text"""
        if language.lower() in ["ur", "urdu"]:
            prompt = f"""
            مندرجہ ذیل متن کی وضاحت صاف اور تعلیمی انداز میں کریں:

            {text}

            تفصیلی وضاحت فراہم کریں جو صارفین کو تصور کو سمجھنے میں مدد کرے۔
            """

            system_message = "آپ ایک تعلیمی اسسٹنٹ ہیں۔ صارفین کو تکنیکی تصورات کو سمجھنے میں مدد کے لیے صاف اور تفصیلی وضاحات فراہم کریں۔ جواب اردو میں دیں۔"
        else:
            prompt = f"""
            Explain the following text in a clear, educational way:

            {text}

            Provide a detailed explanation that helps users understand the concept.
            """

            system_message = "You are an educational assistant. Provide clear, detailed explanations that help users understand technical concepts."

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.4
            )

            result = response.choices[0].message.content.strip()

            # If the user requested Urdu but the response is in English, attempt to translate
            if language.lower() in ["ur", "urdu"]:
                # In a real implementation, we'd ensure the response is in Urdu
                # For now, we'll return the result as is, assuming the model followed the instructions
                pass

            return result
        except Exception as e:
            logger.error(f"Error generating explanation with OpenAI: {str(e)}")
            if language.lower() in ["ur", "urdu"]:
                return "معاف کریں، لیکن میں وضاحت پیدا کرتے وقت ایک خرابی کا سامنا کر رہا ہوں۔ براہ کرم بعد میں دوبارہ کوشش کریں۔"
            else:
                return "I'm sorry, but I encountered an error while generating an explanation. Please try again later."

agent_service = AgentService()