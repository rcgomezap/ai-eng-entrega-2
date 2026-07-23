import asyncio
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# Configuración de logging para una mejor calidad y monitoreo (Arquitectura y Código)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# --- 1. Integridad del Contrato de Datos (Pydantic) ---
class EntityExtraction(BaseModel):
    # Validaciones integradas (min_length) para evitar respuestas vacías
    topic: str = Field(
        description="El tema principal discutido en el texto", 
        min_length=3
    )
    # Python 3.12: Uso de 'list' nativo en lugar de 'typing.List'
    entities: list[str] = Field(
        description="Lista de entidades extraídas del texto"
    )
    sentiment_score: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Puntuación de sentimiento de 0.0 (negativo) a 1.0 (positivo)"
    )

    # Validación semántica: Fuerza que la lista no esté vacía y normaliza los datos
    @field_validator("entities")
    @classmethod
    def check_entities(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("El LLM no extrajo ninguna entidad. La lista no puede estar vacía.")
        # Limpieza de datos: elimina espacios extra y capitaliza cada entidad
        return [entity.strip().title() for entity in v]

# --- 2 y 3. Implementación LCEL y Estrategias de Resiliencia ---
async def run_validated_chain(text: str) -> EntityExtraction | None:
    """
    Ejecuta la cadena de extracción con validación estructurada, reintentos y fallbacks.
    """
    # Definición del modelo principal y un modelo de respaldo (Fallback) más rápido/barato
    primary_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Integración del esquema con .with_structured_output()
    primary_structured = primary_llm.with_structured_output(EntityExtraction)
    fallback_structured = fallback_llm.with_structured_output(EntityExtraction)

    # Resiliencia: Primero reintenta 3 veces en caso de errores transitorios/formato.
    # Si falla definitivamente, hace un 'fallback' al modelo secundario.
    resilient_llm: Runnable = primary_structured.with_retry(
        stop_after_attempt=3
    ).with_fallbacks(
        [fallback_structured]
    )
    
    # Prompt mejorado para darle instrucciones claras al LLM sobre los 3 campos esperados
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analiza el texto. Identifica el tema principal, extrae las entidades clave y evalúa el sentimiento general (de 0.0 a 1.0)."),
        ("human", "{input}")
    ])
    
    # Sintaxis LCEL pura
    chain = prompt | resilient_llm 
    
    try:
        logger.info(f"Procesando texto: '{text}'")
        # Invocación asíncrona
        output: EntityExtraction = await chain.ainvoke({"input": text})
        logger.info("Extracción completada con éxito.")
        return output
    except ValidationError as ve:
        # Captura errores semánticos específicos de Pydantic si fallan todos los reintentos
        logger.error(f"Error de validación de datos (Pydantic):\n{ve}")
    except Exception as e:
        # Captura fallos de red o de la API de OpenAI
        logger.error(f"Error inesperado durante la ejecución: {e}")
    
    return None

# --- 4. Arquitectura y Calidad de Código Inline ---
async def main() -> None:
    sample_text = "LangGraph es una excelente extensión de LangChain para construir agentes cíclicos de manera eficiente."
    result = await run_validated_chain(sample_text)
    
    if result:
        print("\n--- Resultado Validado ---")
        print(f"Tema:        {result.topic}")
        print(f"Entidades:   {result.entities}")
        print(f"Sentimiento: {result.sentiment_score}")

if __name__ == "__main__":
    asyncio.run(main())