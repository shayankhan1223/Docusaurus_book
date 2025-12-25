from typing import List, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def translate_text(self, text: str, target_language: str = "ur", source_language: str = "en") -> str:
        """Translate text from source language to target language using AI"""
        try:
            system_prompt = ""
            if target_language.lower() in ["ur", "urdu"]:
                system_prompt = "You are a professional translator. Translate the given text from English to Urdu. Maintain the meaning and context accurately. Use proper Urdu script and grammar. Respond only with the translated text."
            else:
                # Default to English if target language is not Urdu
                return text

            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate this to {target_language}:\n\n{text}"}
                ],
                max_tokens=1000,
                temperature=0.3
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error translating text to {target_language}: {str(e)}")
            return text  # Return original text if translation fails

    async def translate_documentation_content(self, content: Dict, target_language: str = "ur") -> Dict:
        """Translate documentation content fields to target language"""
        if target_language.lower() not in ["ur", "urdu"]:
            return content  # Only translate to Urdu for now

        translated_content = content.copy()

        # Translate title if it exists
        if "title" in translated_content:
            translated_content["title"] = await self.translate_text(
                translated_content["title"],
                target_language
            )

        # Translate content if it exists
        if "content" in translated_content:
            translated_content["content"] = await self.translate_text(
                translated_content["content"],
                target_language
            )

        # Translate other fields if they exist
        if "content_preview" in translated_content:
            translated_content["content_preview"] = await self.translate_text(
                translated_content["content_preview"],
                target_language
            )

        return translated_content

# Global instance
translation_service = TranslationService()