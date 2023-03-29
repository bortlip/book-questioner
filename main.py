import sys
import time

from pathlib import Path
from nltk.tokenize import sent_tokenize, word_tokenize
from my_module.memory import memory
from my_module.TextUtilities import split_text_into_sections_by_words, split_text_into_sections_by_sentences
from my_module.agent import GPT35Agent
from my_module.agent_settings import AgentSettings

def main():
    print("loading memory")

    mem = memory()

    print("Reading file")
    file_path = Path('thebible.txt')
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        sys.exit(1)

    word_count = len(word_tokenize(content))
    print(f"Text word count: {word_count}")

    print("Splitting into sections")
    sections = split_text_into_sections_by_words(content, 100)

    print("Adding to memory")
    for section in sections:
        if section:
            mem.add(section)

    summaryAgentPrompt = "You are a summarizer. You will summarize the provided text."

    message = f"""
    You answer questions based on a provided context.
    You answer questions as thoroughly as possible using only the provided context.
    If the context doesn't provide an answer, you indicate that.
    You provide citations inline with the answer text.
    CITE EVERYTHING!!!
    """

    settings = AgentSettings(summaryAgentPrompt, "agent", 2000, 3000, 0.0, 3900)
    agent = GPT35Agent(settings)

    while True:

        user_input = input(">")

        print("searching")

        results = mem.search_word_length(user_input, 2500)

        context = "".join(results)

        question = f"""
    Answer the question as thoroughly as possible using only the provided context.
    If the context doesn't provide an answer, indicate that.
    Provide citations inline with the answer text.
    Question: 
    {user_input}
    context: 
    {context}
    """

        agent.add_user_message(question)

        print("getting response")

        response = agent.step_session()

        print("\n----\n")
        print(response)

if __name__ == "__main__":
    main()