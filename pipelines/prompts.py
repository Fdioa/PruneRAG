def get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the energy range of pp III neutrinos?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_math_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_code_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Find the minimum number of vertices in a Steiner tree that includes all specified vertices in a given tree.\"\n"
        "Assistant thinking steps:\n"
        "- I need to understand what a Steiner tree is and how to compute the minimum number of vertices required to include all specified vertices in a given tree.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Minimum Steiner Tree problem in trees<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

    
def get_singleqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention several URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/first_nobel_physics.html<|end_url|>\n\n"
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_multiqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>voice actor of Lara Croft<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results provide names of voice actors for Lara Croft. I need to confirm if Alice David is one of them.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention the company that developed the video game featuring Alice David as Lara Croft.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/lara_croft_voice_actor.html, http://example.com/game_developer.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_gpqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the energy range of pp III neutrinos?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/ppIII_neutrino.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_math_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s math-related question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/integration_e_x_squared.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_code_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s programming-related question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <|begin_url|> url1, url2, ... <|end_url|>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <|begin_full_page|> ...full page content... <|end_full_page|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do I implement a binary search algorithm in Python?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up the implementation details of binary search in Python.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>binary search algorithm implementation in Python<|end_search_query|>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<|begin_search_result|> ...search results without full page... <|end_search_result|>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<|begin_url|>http://example.com/python_binary_search.html<|end_url|>\n\n" 
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<|begin_full_page|> ...full page content... <|end_full_page|>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- Use <|begin_url|> to request full page content and end with <|end_url|>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_naive_rag_instruction(question, documents):
    return (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
        "Question:\n"
        f"{question}\n"
        "Documents:\n"
        f"{documents}\n"
    )



def get_task_instruction_openqa(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_math(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following math question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following math question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_multi_choice(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following multiple-choice question. '
            'You should provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'llama':
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Provide your final choice in the format \\boxed{YOUR_CHOICE}. Your final choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_code(question, question_title=None, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Generate a correct Python program that passes all tests for the given problem. '
            'You should provide your final code within a Python code block using triple backticks (```python\n'
            'YOUR_CODE\n'
            '```).\n\n'
            f'Problem Title: {question_title}\n\n'
            f'Problem Statement:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. '
            f'You should think step by step to solve it.\n\nQuestion:\n{question}\n\n'
            'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n\n'
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return user_prompt

def get_react_examples():
    user_prompt = (
        "\nQuestion: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?\nThought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.\nAction 1: Search[Colorado orogeny]\nObservation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.\nThought 2: It does not mention the eastern sector. So I need to look up eastern sector.\nAction 2: Lookup[eastern sector]\nObservation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.\nThought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.\nAction 3: Search[High Plains]\nObservation 3: High Plains refers to one of two distinct land regions:\nThought 4: I need to instead search High Plains (United States).\nAction 4: Search[High Plains (United States)]\nObservation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]\nThought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.\nAction 5: Finish[1,800 to 7,000 ft]\nQuestion: Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?\nThought 1: The question simplifies to \"The Simpsons\" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.\nAction 1: Search[Milhouse]\nObservation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.\nThought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up \"named after\".\nAction 2: Lookup[named after]\nObservation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. \nThought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.\nAction 3: Finish[Richard Nixon]\nQuestion: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\nThought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.\nAction 1: Search[Adam Clayton Powell]\nObservation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].\nThought 2: To find the documentary, I can search Adam Clayton Powell (film).\nAction 2: Search[Adam Clayton Powell (film)]\nObservation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.\nThe film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.\nThought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.\nAction 3: Finish[The Saimaa Gesture]\nQuestion: What profession does Nicholas Ray and Elia Kazan have in common?\nThought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.\nAction 1: Search[Nicholas Ray]\nObservation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 \u2013 June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.\nThought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.\nAction 2: Search[Elia Kazan]\nObservation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.\nThought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.\nAction 3: Finish[director, screenwriter, actor]\nQuestion: Which magazine was started first Arthur's Magazine or First for Women?\nThought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.\nAction 1: Search[Arthur's Magazine]\nObservation 1: Arthur's Magazine (1844-\u0080\u00931846) was an American literary periodical published in Philadelphia in the 19th century. \nThought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.\nAction 2: Search[First for Women]\nObservation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. \nThought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.\nAction 3: Finish[Arthur's Magazine]\nQuestion: Were Pavel Urysohn and Leonid Levin known for the same type of work?\nThought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.\nAction 1: Search[Pavel Urysohn]\nObservation 1: Pavel Samuilovich Urysohn (February 3, 1898 \u00e2\u0080\u0093 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.\nThought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.\nAction 2: Search[Leonid Levin]\nObservation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. \nThought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. \nAction 3: Finish[yes]\n"
    )
    return user_prompt