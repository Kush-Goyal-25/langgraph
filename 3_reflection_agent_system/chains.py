import datetime

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


# Define Pydantic models for tools
class AnswerQuestion(BaseModel):
    """Model for answering a question."""

    answer: str = Field(description="The detailed answer to the question (~250 words).")


class ReviseAnswer(BaseModel):
    """Model for revising a previous answer."""

    revised_answer: str = Field(description="The revised answer (~250 words).")
    references: list[str] = Field(description="List of reference URLs.")


# Initialize parser
pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion, ReviseAnswer])

# Actor Agent Prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI researcher.
                Current time: {time}

                1. {first_instruction}
                2. Reflect and critique your answer. Be severe to maximize improvement.
                3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.
            """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

# First responder prompt
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=500,
)

# First responder chain with parser
first_responder_chain = (
    first_responder_prompt_template
    | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
    | pydantic_parser
)

# Revisor section
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor_chain = (
    actor_prompt_template.partial(first_instruction=revise_instructions)
    | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
    | pydantic_parser
)

# Invoke the chain
response = first_responder_chain.invoke(
    {"messages": [HumanMessage("AI Agents taking over content creation")]}
)

print(response)
