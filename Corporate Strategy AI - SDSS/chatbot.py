# =====================================================
# IMPORT LIBRARY
# =====================================================
import os

from groq import Groq
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv(
    "groq_api"
)

# =====================================================
# GROQ CLIENT
# =====================================================

client = Groq(
    api_key=GROQ_API_KEY
)
# =====================================================
# CHATBOT FUNCTION
# =====================================================
def ask_ai(prompt):

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[

            {
                "role": "system",
                "content": """
                You are an AI Operational Efficiency Assistant.
                Analyze operational data and provide strategic recommendations.
                """
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        temperature=0.7

    )

    return response.choices[0].message.content