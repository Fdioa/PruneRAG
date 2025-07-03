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

def get_react_examples():
    user_prompt = (
        "Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?\nThought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.\nAction 1: Search[Pavel Urysohn]\nObservation 1: Pavel Samuilovich Urysohn (February 3, 1898 \u00e2\u0080\u0093 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.\nThought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.\nAction 2: Search[Leonid Levin]\nObservation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. \nThought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. \nAction 3: Finish[yes]\n"
    )
    return user_prompt

def get_subqueries_qwen3_8b():
    user_prompt = (
        '''
        Your goal is to answer or decompose the 'Query' using the provided 'Query_context' and 'Parent_query'.

        1.  **Context Assessment**: Determine if the 'Query_context' is helpful for the 'Query'.
            * If not helpful, ignore it.
            * If helpful, use its information.
        2. **Parent Query**: The 'Parent_query' is the query that led to the current 'Query'. It may provide additional constraints.
        3.  **Output Determination**:
            * **Direct Answer**: If the 'Query_context' allows for a direct answer to the 'Query' without decomposition, output in the format: `{{\"answer\": \"...\"}}`.If you are not 100% sure the answer is correct, please do not answer.
            * **Decomposition**: If the 'Query' can't answer directly and the 'Query' requires decomposition into two logically related sub-queries to deduce the answer, output in the format: `{{\"subquery1\": \"...\", \"subquery2\": \"...\"}}`. These two sub-queries must logically lead to the original query's answer.
            * **No Answer**: If the 'Query_context' does not provide any helpful information for the 'Query' and there is no need to decompose further, output in the format: `{{\"subquery1\": \"A key entity in the Query\", \"subquery2\": \"A key entity in the Query\"}}`.
        
        Please output your final response strictly in accordance with the above JSON format after thinking about it.


        Example 1 (Direct Answer):
        Query_context:  Doc 1:  First for Women is a woman's magazine published by Bauer Media Group in the USA.The magazine was started in 1989.It is based in Englewood Cliffs, New Jersey.In 2011 the circulation of the magazine was 1,310,696 copies. 
                        Doc 2: Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others.In May 1846 it was merged into "Godey's Lady's Book".\n
        Parent_query of the Query: Which magazine was started first Arthur's Magazine or First for Women?
        Query: When was the Arthur's Magazine started?
        Output: {{\"answer\": \"1844\"}}

        Example 2 (Decomposition):
        Query_context:  Doc 1:A Flame in My Heart is a 1987 French- Swiss drama film directed by Alain Tanner.
                        Doc 2:\"A Hole in My Heart\"\nA Hole in My Heart A Hole in My Heart () is a 2004 Swedish experimental drama film written and directed by Lukas Moodysson, starring Thorsten Flinck, Sanna Bråding, Björn Almroth and Goran Marjanovic. The story revolves around a man who makes a pornographic film in his apartment with a friend and an attention-seeking starlet, while his teenage son stays in his room and listens to ambient noise music. The film is notable for its explicit imagery, including close-ups of vaginal reconstruction surgery, an anal sex scene without the use of lubrication, a masturbation scene with a toothbrush.
                        Doc 3:\"A Hole in My Heart\"\na 41% approval percentage based on 17 reviews at Rotten Tomatoes. A Hole in My Heart A Hole in My Heart () is a 2004 Swedish experimental drama film written and directed by Lukas Moodysson, starring Thorsten Flinck, Sanna Bråding, Björn Almroth and Goran Marjanovic. The story revolves around a man who makes a pornographic film in his apartment with a friend and an attention-seeking starlet, while his teenage son stays in his room and listens to ambient noise music. The film is notable for its explicit imagery, including close-ups of vaginal reconstruction surgery, an anal sex scene without.
        Query: What is the birhday of the director of A Flame In My Heart?
        Parent_query of the Query:  Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
        Output: {{\"subquery1\": \"Who is Alain Tanner?\", \"subquery2\": \"What is the birhday of Alain Tanner?\"}}

        Example 3 (No Answer):
        Query_context:  Doc 1:"B.C. Butcher" B.C. Butcher B.C. Butcher is a 2016 American horror comedy film directed by 17-year-old Kansas Bowling about a tribe of cavewomen being stalked by a prehistoric monster. It has been dubbed as \"\"the first prehistoric slasher film\"\". It was released in January 2016 by Troma Entertainment. A tribe of cavewomen sacrifice one of their members after it is revealed she is having an affair with the tribe leader's man (Kato Kaelin). They leave her body in the wilderness and it is discovered by a prehistoric beast who falls in love with the dead cavewoman and vows to avenge her death.
                        Doc 2:\"Night Warning\"\nMovies of the 1980s\"\", John Kenneth Muir rated it 3.5/4 stars. Muir called it \"\"a true gem of the decade\"\" and \"\"the 1980s most twisted, bizarre cinematic vision of motherhood\"\". Night Warning Night Warning (also known as Butcher, Baker, Nightmare Maker) is a 1982 American exploitation horror film directed by William Asher, and starring Susan Tyrrell, Jimmy McNichol, Julia Duffy, and Bo Svenson. Framed as a contemporary Oedipus tale, the plot focuses on a teenager who, raised by his neurotic aunt, finds himself at the center of a murder investigation after she stabs a man to death in their house.
                        Doc 3:\"The Butcher Boy (1997 film)\"\nThe Butcher Boy (1997 film) The Butcher Boy is a 1997 Irish-American tragicomic drama film adapted to film by Neil Jordan and Patrick McCabe from McCabe's 1992 novel of the same name. Set in the early 1960s, \"\"The Butcher Boy\"\" is about Francie Brady (Eamonn Owens), a 12-year-old boy who retreats into a violent fantasy world to escape the reality of his dysfunctional family; as his circumstances worsen, his sanity deteriorates and he begins acting out, with increasing brutality. The film won the Silver Bear for Best Director at the 48th Berlin International Film Festival in 1998 and a Special.
        Parent_query of the Query: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
        Query: Who is the director of Butcher, Baker, Nightmare Maker?
        Output: {{\"subquery1\": \"Butcher, Baker, Nightmare Maker\", \"subquery2\": \"Direct of Butcher, Baker, Nightmare Maker\"}}

        Query_context: {context}
        Parent_query of the Query: {parent_query}
        Query: {query}
        Output:
        '''
    )
    return user_prompt


def get_subqueries_qwen3_8b_first():
    user_prompt = (
        '''
        Your goal is to decompose the 'Query' using the provided 'Query_context'.

        1.  **Context Assessment**: Determine if the 'Query_context' is helpful for the 'Query'.
            * If not helpful, ignore it.
            * If helpful, use its information.
        2.  **Output Determination**:
            * **Decomposition**: Please decompose the 'Query' into two logically related sub-queries to deduce the answer, output in the format: `{{\"subquery1\": \"...\", \"subquery2\": \"...\"}}`. These two sub-queries must logically lead to the original query's answer.

        Please output your final response strictly in accordance with the above JSON format after thinking about it.

        Example (Decomposition):
        Query_context:  Doc 1:A Flame in My Heart is a 1987 French- Swiss drama film directed by Alain Tanner.
                        Doc 2:\"A Hole in My Heart\"\nA Hole in My Heart A Hole in My Heart () is a 2004 Swedish experimental drama film written and directed by Lukas Moodysson, starring Thorsten Flinck, Sanna Bråding, Björn Almroth and Goran Marjanovic. The story revolves around a man who makes a pornographic film in his apartment with a friend and an attention-seeking starlet, while his teenage son stays in his room and listens to ambient noise music. The film is notable for its explicit imagery, including close-ups of vaginal reconstruction surgery, an anal sex scene without the use of lubrication, a masturbation scene with a toothbrush.
                        Doc 3:\"A Hole in My Heart\"\na 41% approval percentage based on 17 reviews at Rotten Tomatoes. A Hole in My Heart A Hole in My Heart () is a 2004 Swedish experimental drama film written and directed by Lukas Moodysson, starring Thorsten Flinck, Sanna Bråding, Björn Almroth and Goran Marjanovic. The story revolves around a man who makes a pornographic film in his apartment with a friend and an attention-seeking starlet, while his teenage son stays in his room and listens to ambient noise music. The film is notable for its explicit imagery, including close-ups of vaginal reconstruction surgery, an anal sex scene without.
        Query: What is the birhday of the director of A Flame In My Heart or Butcher?
        Output: {{\"subquery1\": \"Who is Alain Tanner?\", \"subquery2\": \"What is the birhday of  Alain Tanner?\"}}

        Query_context: {context}
        Query: {query}
        Output:
        '''
    )

    return user_prompt


def get_final_answer_qwen3_8b():
    user_prompt = (
            '''
            Please answer the following question.
            You can use the information in the inference tree below.If there are logical errors in the inference tree, please correct them based on your knowledge or do not use this wrong information as the basis for your response.
            Your answer should be formatted as \\boxed{{YOUR_ANSWER}}. 

            
            ## Inference Tree
            {context}

            ## Question
            {question}'''
        )

    return user_prompt



def get_subqueries_llama3_8b():
    user_prompt = (
        '''
Your task is to answer or decompose the 'Query' using the provided 'Query_context' and 'Parent_query'.

**Instructions:**

1.  **Analyze Context & Parent Query**: Evaluate if 'Query_context' and 'Parent_query' are helpful for the 'Query'.
2.  **Determine Output Type**:
    * **Direct Answer**: If 'Query_context' contains the **exact and verifiable answer** to the 'Query' (without needing decomposition), output: {{"answer": "..."}}. **Only answer if you 100% certain.**
    * **Decomposition**: If the 'Query' cannot be directly answered and requires breaking down into two **logically related sub-queries** that together **lead to the original query's answer**, output: {{"subquery1": "...", "subquery2": "..."}}.
    * **No Answer**: If 'Query_context' is **unhelpful** and decomposition is **not necessary or possible** (e.g., no key entities to decompose), output: {{"subquery1": "A key entity in the Query", "subquery2": "A key entity in the Query"}}.
3.  **Rules To Follow**: THE SUBQUERIES OR ANSWER MUST BE IN JSON FORMAT.Your output must be ONLY a JSON object, with no other introductory or explanatory text.The value of "subquery1", "subquery2" and "answer" in the json format MUST be a STRING.
4.  **Important Note**: Your JSON output must be wrapped in a Markdown code block like this:
```json
{{"subquery1": "...","subquery2": "..."}}
```
```json
{{"answer": "..."}}
```

**Example 1 (Direct Answer):**

Query_context: Doc 1: Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.
Parent_query of the Query: Which magazine was started first Arthur's Magazine or First for Women?
Query: When was the Arthur's Magazine started?
Output: 
``` json
{{"answer": "1844"}}
```

**Example 2 (Decomposition):**
Query_context:  Doc 1: A Flame in My Heart is a 1987 French- Swiss drama film directed by Alain Tanner.
Parent_query of the Query: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Query: What is the birhday of the director of A Flame In My Heart?
Output: 
``` json
{{"subquery1": "Who is Alain Tanner?", "subquery2": "What is the birhday of Alain Tanner?"}}
```

**Example 3 (No Answer):**
Query_context: ...
Parent_query of the Query: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Query: Who is the director of Butcher, Baker, Nightmare Maker?
Output: 
``` json 
{{"subquery1": "Butcher, Baker, Nightmare Maker", "subquery2": "Direct of Butcher, Baker, Nightmare Maker"}}
```

Query_context: {context}\n
Parent_query of the Query: {parent_query}\n
Query: {query}\n
Output:
        '''
    )
    return user_prompt


def get_subqueries_llama3_8b_first():
    user_prompt = (
        '''
        Your task is to decompose the 'Query' into sub-queries. Refer to the 'Query_context' as needed.

        **Instructions:**
        1.  **Decompose Query**: Decompose the 'Query' into two **logically related** sub-queries. Both sub-queries must collectively **lead to the original query's answer**.
        2.  **Output Format**: Output your final answer strictly in the following JSON format: {{"subquery1": "...", "subquery2": "..."}}.
        3.  **Rules To Follow**: THE SUBQUERIES MUST BE IN JSON FORMAT.Your output must be ONLY a JSON object, with no other introductory or explanatory text.The value of "subquery1" and "subquery2" in the json format MUST be a STRING.
        4.  **Important Note**: Your JSON output must be wrapped in a Markdown code block like this:
        ```json
        {{"subquery1": "...","subquery2": "..."}}
        ```

        **Example:**
        Query_context:...
        Query: Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?
        Output: 
        ``` json 
        {{\"subquery1\": \"Who is the director of Move (1970 Film)?\", \"subquery2\": \"Who is the director of Méditerranée (1963 Film)\"}}
        ```

        **Your Task:**
        Query_context: {context}
        Query: {query}
        Output:
        '''
        
    )

    return user_prompt


def get_final_answer_llama3_8b():
    user_prompt = (
            '''
            ## Inference Tree
            {context}

            Please answer the following question.
            You can use the helpful information in the above Inference Tree.
            Your final answer must be formatted as \\boxed{{YOUR_ANSWER}}.And YOUR_ANSWER should be a NON-SENTENTIAL answer.
            For example, Question: What is the capital of France? Answer: \\boxed{{Paris}}.

            ## Question
            {question}

            Answer: 

            '''
        )

    return user_prompt