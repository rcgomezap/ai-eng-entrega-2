import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatOpenAI(model='openrouter/free', base_url=os.getenv('OPENAI_BASE_URL'), temperature=float(os.getenv("TEMPERATURE", "0.7")))
parser = StrOutputParser()

question_chain = (
    ChatPromptTemplate.from_template(
    "Answer this consicely in the language from the inquiry: {question}"
    )
    | model
    | parser
)

async def main():
    question = {"question": "Que es la mecanica cuantica?"}
    response = await question_chain.ainvoke(question)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
