import os

from dotenv import load_dotenv

# Removed unused imports: initialize_agent, tool, TavilySearchResults, PromptValue
# Kept HumanMessage as it's used conceptually and in basic.py
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables (ensure you have GOOGLE_API_KEY in your .env file)
load_dotenv()

# Check if the API key is loaded (optional but good practice)
if os.getenv("GOOGLE_API_KEY") is None:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")
    # You might want to exit or raise an error here depending on your needs
    # exit("Please set the GOOGLE_API_KEY environment variable.")

llm = ChatGoogleGenerativeAI(
    # Use a valid and available model name
    model="gemini-1.5-flash-latest",  # Or "gemini-pro" etc.
    temperature=0.7,
    # Consider removing max_tokens if you want the model to decide completion length,
    # or keep it if you need strict control. 500 might be small for detailed reflection.
    # max_tokens=500,
    convert_system_message_to_human=True,  # Often helpful for Gemini models
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts."
            " Look at the conversation history for the user request and previous critique/attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Analyze the last AI-generated tweet in the conversation history."
            " Generate constructive critique and detailed recommendations for improvement."
            " Consider aspects like length, engagement, clarity, hashtags, tone, and virality potential."
            " Provide specific suggestions.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Chains remain the same LCEL structure
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

# You can optionally test the chains here
if __name__ == "__main__":
    print("Testing generation chain...")
    test_gen_result = generation_chain.invoke(
        {"messages": [HumanMessage(content="Tweet about the future of AI in coding")]}
    )
    print(test_gen_result)
    print("\nTesting reflection chain...")
    test_reflect_result = reflection_chain.invoke(
        {
            "messages": [
                HumanMessage(content="Tweet about the future of AI in coding"),
                test_gen_result,
            ]
        }
    )
    print(test_reflect_result)
