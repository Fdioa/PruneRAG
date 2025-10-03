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
```[Helpful information]```
- **If the web pages do not provide any helpful information for current search query:** Output the following text.
**Final Information**
```No helpful information found.```


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


def get_native_instruction(multi_choice=False):
    if multi_choice:
        user_prompt = (
            '''
            Answer the following multiple-choice question:
            IMPORTANT: You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}, and YOUR_ANSWER should be one of the letters A, B, C, or D, DO NOT include any answer content.
            For example, Question: What is the capital of France?\n(A) Paris \n(B) London \n(C) Berlin \n(D) Dubai \n Answer: \\boxed{{A}}.
            Question: {question}
            Answer:
        '''
        )    
    else:
        user_prompt = (
            '''
                Answer the following question:
                IMPORTANT: You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.
                For example, Question: What is the capital of France? Answer: \\boxed{{Paris}}.
                Question: {question}
                Answer:'''
        )
    return user_prompt

def get_rag_instruction(multi_choice=False):
    if multi_choice:
        user_prompt = (
        '''
            Answer the following multiple-choice question:
            You can use the following documents to help you answer the question.
            Documents: {context}
            IMPORTANT: You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}, and YOUR_ANSWER should be one of the letters A, B, C, or D, DO NOT include any answer content.
            For example, Question: What is the capital of France?\n(A) Paris \n(B) London \n(C) Berlin \n(D) Dubai \n Answer: \\boxed{{A}}.
            Question: {question}
            Answer:
        '''
    )
    else:
        user_prompt = (
            '''
                Answer the following question:
                You can use the following documents to help you answer the question.
                Documents: {context}
                IMPORTANT: You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.
                For example, Question: What is the capital of France? Answer: \\boxed{{Paris}}.
                Question: {question}
                Answer:
            '''
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
You are given a query along with its parent question and optional context. Your task is to select the correct action type based on the following rules:
### Available Action Types:
1. **Direct Answer**  
If the `Query_context` contains a clear and verifiable answer to the `Query`, respond as:  
```json
{{"type": "answer", "answer": "..."}}
```
2. **Decomposition**
If the Query cannot be directly answered,please split the query into two logically related subqueries that together answer the parent query, respond as:
```json
{{"type": "decomposition", "subquery1": "...", "subquery2": "..."}}
```
3. **Entity Extraction**
If the query contains only one entity information that needs to be clarified, extract the entity and express it as two entities from different perspectives, respond as:
```json
{{"type": "entity", "entity1": "...", "entity2": "..."}}
```
### Rules:
Ignore information not relevant to the query in the Query_context.
Only use "answer" when you are **100% sure** the answer is directly supported by the context.
Ensure subquery1, subquery2, entity1, entity2, and answer in json format are all string values.
The output must be a single JSON object inside a markdown code block.
Please think carefully before making your choice.
### Examples:
**Example 1 (Direct Answer):**
Query_context: Doc 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
Parent_query of the Query: Which magazine was started first Arthur's Magazine or First for Women?
Query: When was the Arthur's Magazine started?
Output: 
``` json
{{"type": "answer", "answer": "1844"}}
```
**Example 2 (Decomposition):**
Query_context:  Doc 1: A Flame in My Heart is a 1987 French- Swiss drama film directed by Alain Tanner.
Parent_query of the Query: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Query: What is the birhday of the director of A Flame In My Heart?
Output: 
``` json
{{"type": "decomposition", "subquery1": "Who is Alain Tanner?", "subquery2": "What is the birhday of Alain Tanner?"}}
```
**Example 3 (Entity Extraction):**
Query_context: ...
Parent_query of the Query: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Query: Who is the director of Butcher, Baker, Nightmare Maker?
Output: 
``` json 
{{"type": "entity", "entity1": "Butcher, Baker, Nightmare Maker", "entity2": "Director of Butcher, Baker, Nightmare Maker"}}
```
### Your Task:
Query_context: {context}
Parent_query of the Query: {parent_query}
Query: {query}
Output:
        '''
    )
    return user_prompt

def get_subqueries_qwen3_8b_wo_ans():
    user_prompt = (
'''
You are given a query along with its parent question and optional context. Your task is to select the correct action type based on the following rules:
### Available Action Types:
1. **Decomposition**
If the Query cannot be directly answered,please split the query into two logically related subqueries that together answer the parent query, respond as:
```json
{{"type": "decomposition", "subquery1": "...", "subquery2": "..."}}
```
2. **Entity Extraction**
If the query contains only one entity information that needs to be clarified, extract the entity and express it as two entities from different perspectives, respond as:
```json
{{"type": "entity", "entity1": "...", "entity2": "..."}}
```

### Rules:
Ignore information not relevant to the query in the Query_context.
Ensure subquery1, subquery2, entity1, entity2 in json format are all string values.
The output must be a single JSON object inside a markdown code block.
Please think carefully before making your choice.


### Your Task:
Query_context: {context}
Parent_query of the Query: {parent_query}
Query: {query}
Output:
        '''
    )
    return user_prompt

def get_subqueries_qwen3_8b_wo_ent():
    user_prompt = (
'''
You are given a query along with its parent question and optional context. Your task is to select the correct action type based on the following rules:
### Available Action Types:
1. **Direct Answer**  
If the `Query_context` contains a clear and verifiable answer to the `Query`, respond as:  
```json
{{"type": "answer", "answer": "..."}}
```
2. **Decomposition**
If the Query cannot be directly answered,please split the query into two logically related subqueries that together answer the parent query, respond as:
```json
{{"type": "decomposition", "subquery1": "...", "subquery2": "..."}}
```

### Rules:
Ignore information not relevant to the query in the Query_context.
Only use "answer" when you are **100% sure** the answer is directly supported by the context.
Ensure subquery1, subquery2 and answer in json format are all string values.
The output must be a single JSON object inside a markdown code block.
Please think carefully before making your choice.


### Your Task:
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
You are given a query along with its parent question and optional context. Your task is to decompose the query into two logically related sub-queries.

If the Query cannot be directly answered but can be split into two logically related subqueries that together answer the parent query, respond as:
```json
{{"type": "decomposition", "subquery1": "...", "subquery2": "..."}}
```

### Rules:
Ignore information not relevant to the query in the Query_context.
Ensure subquery1, subquery2 are all string values.
The output must be a single JSON object inside a markdown code block.
Do not provide any explanation or commentary outside the JSON.

### Your Task:
Query_context: {context}
Query: {query}
Output:
'''
    )

    return user_prompt








def get_final_answer_qwen3_8b():
    user_prompt = (
            '''
### Inference Tree:
{context}

Please answer the following question.
You can use the helpful information in the above Inference Tree.
Your final answer must be formatted as \\boxed{{YOUR_ANSWER}}.And YOUR_ANSWER should be a NON-SENTENTIAL answer.
For example, Question: What is the capital of France? Answer: \\boxed{{Paris}}.

### Question:
{question}

### Answer: 
'''
        )

    return user_prompt

def get_final_answer_qwen3_8b_multi_choice():
    user_prompt = (
            '''
### Inference Tree:
{context}

Please answer the following multiple-choice question.
You can use the helpful information in the above Inference Tree.
Your final answer must be formatted as \\boxed{{YOUR_ANSWER}}.And YOUR_ANSWER should be one of the letters A, B, C, or D, DO NOT include any answer content.
For example, Question: What is the capital of France?\n(A) Paris \n(B) London \n(C) Berlin \n(D) Dubai \n Answer: \\boxed{{A}}.

### Question:
{question}

### Answer: 
'''
        )

    return user_prompt

def get_subqueries_llama3_8b_first():
    user_prompt = (
        '''
You are given a query along with its parent question and optional context. Your task is to decompose the query into two logically related sub-queries.

If the Query cannot be directly answered but can be split into two logically related subqueries that together answer the parent query, respond as:
```json
{{"type": "decomposition", "subquery1": "...", "subquery2": "..."}}
```
### Rules:
Ignore information not relevant to the query in the Query_context.
Ensure subquery1, subquery2 are all string values.
The output must be a single JSON object inside a markdown code block.
Do not provide any explanation or commentary outside the JSON.

### Example:
Query_context: ...
Query: ...
Output:
```json
{{"type": "decomposition", "subquery1": "...", "subquery2": "..."}}
```

### Your Task:
Query_context: {context}
Query: {query}
Output:
'''
    )

    return user_prompt

def get_final_answer_llama3_8b(multi_choice=False):
    if multi_choice:
        user_prompt = (
                '''
    ### Inference Tree:
    {context}
    Please answer the following multiple-choice question.
    You can use the helpful information in the above Inference Tree.
    Your final answer must be formatted as \\boxed{{YOUR_ANSWER}}.And YOUR_ANSWER should be one of the letters A, B, C, or D, DO NOT include any answer content.
    For example, Question: What is the capital of France?\n(A) Paris \n(B) London \n(C) Berlin \n(D) Dubai \n Answer: \\boxed{{A}}.

    ### Question:
    {question}
    ### Answer:
    '''
            )

    else:
        user_prompt = (
                '''
    ### Inference Tree:
    {context}

    Please answer the following question.
    You can use the helpful information in the above Inference Tree.
    Your final answer must be formatted as \\boxed{{YOUR_ANSWER}}.And YOUR_ANSWER should be a NON-SENTENTIAL answer.
    For example, Question: What is the capital of France? Answer: \\boxed{{Paris}}.

    ### Question:
    {question}

    ### Answer: 
    '''
            )

    return user_prompt

def get_final_answer_llama3_8b_multi_choice():
    user_prompt = (
            '''
### Inference Tree:
{context}

Please answer the following multiple-choice question.
You can use the helpful information in the above Inference Tree.
Your final answer must be formatted as \\boxed{{YOUR_ANSWER}}.And YOUR_ANSWER should be one of the letters A, B, C, or D, DO NOT include any answer content.
For example, Question: What is the capital of France?\n(A) Paris \n(B) London \n(C) Berlin \n(D) Dubai \n Answer: \\boxed{{A}}.

### Question:
{question}

### Answer: 
'''
        )

    return user_prompt

def get_subqueries_llama3_8b():
    user_prompt = (
'''
You are given a query along with its parent question and optional context. Your task is to select the correct action type based on the following rules:
### Available Action Types:
1. **Direct Answer**  
If the `Query_context` contains a clear and verifiable answer to the `Query`, respond as:  
```json
{{"type": "answer", "answer": "..."}}
```
2. **Decomposition**
If the Query cannot be directly answered but can be split into two logically related subqueries that together answer the parent query, respond as:
```json
{{"type": "decomposition", "subquery1": "...", "subquery2": "..."}}
```
3. **Entity Extraction**
If the Query lacks sufficient context to be answered or decomposed, but contains key identifiable entities, extract them as:
```json
{{"type": "entity", "entity1": "...", "entity2": "..."}}
```
### Rules:
Only use "answer" when you are 100% sure the answer is directly supported by the context.
Ignore information not relevant to the query in the Query_context.
Ensure subquery1, subquery2, entity1, entity2, and answer are all string values.
The output must be a single JSON object inside a markdown code block.
Do not provide any explanation or commentary outside the JSON.
### Examples:
**Example 1 (Direct Answer):**
Query_context: Doc 1: Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.
Parent_query of the Query: Which magazine was started first Arthur's Magazine or First for Women?
Query: When was the Arthur's Magazine started?
Output: 
``` json
{{"type": "answer", "answer": "1844"}}
```
**Example 2 (Decomposition):**
Query_context:  Doc 1: A Flame in My Heart is a 1987 French- Swiss drama film directed by Alain Tanner.
Parent_query of the Query: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Query: What is the birhday of the director of A Flame In My Heart?
Output: 
``` json
{{"type": "decomposition", "subquery1": "Who is Alain Tanner?", "subquery2": "What is the birhday of Alain Tanner?"}}
```
**Example 3 (Entity Extraction):**
Query_context: ...
Parent_query of the Query: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Query: Who is the director of Butcher, Baker, Nightmare Maker?
Output: 
``` json 
{{"type": "entity", "entity1": "Butcher, Baker, Nightmare Maker", "entity2": "Director of Butcher, Baker, Nightmare Maker"}}
```
### Your Task:
Query_context: {context}
Parent_query of the Query: {parent_query}
Query: {query}
Output:

''')

    return user_prompt

def get_subqueries_llama3_8b_wo_ent():
    user_prompt = (
'''
You are given a query along with its parent question and optional context. Your task is to select the correct action type based on the following rules:
### Available Action Types:
1. **Direct Answer**  
If the `Query_context` contains a clear and verifiable answer to the `Query`, respond as:  
```json
{{"type": "answer", "answer": "..."}}
```
2. **Decomposition**
If the Query cannot be directly answered but can be split into two logically related subqueries that together answer the parent query, respond as:
```json
{{"type": "decomposition", "subquery1": "...", "subquery2": "..."}}
```
### Rules:
Only use "answer" when you are 100% sure the answer is directly supported by the context.
Ignore information not relevant to the query in the Query_context.
Ensure subquery1, subquery2 and answer are all string values.
The output must be a single JSON object inside a markdown code block.
Do not provide any explanation or commentary outside the JSON.
### Examples:
**Example 1 (Direct Answer):**
Query_context: Doc 1: Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.
Parent_query of the Query: Which magazine was started first Arthur's Magazine or First for Women?
Query: When was the Arthur's Magazine started?
Output: 
``` json
{{"type": "answer", "answer": "1844"}}
```
**Example 2 (Decomposition):**
Query_context:  Doc 1: A Flame in My Heart is a 1987 French- Swiss drama film directed by Alain Tanner.
Parent_query of the Query: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Query: What is the birhday of the director of A Flame In My Heart?
Output: 
``` json
{{"type": "decomposition", "subquery1": "Who is Alain Tanner?", "subquery2": "What is the birhday of Alain Tanner?"}}
```
### Your Task:
Query_context: {context}
Parent_query of the Query: {parent_query}
Query: {query}
Output:

''')
    
    return user_prompt

def get_subqueries_llama3_8b_wo_ans():
    user_prompt = (
'''
You are given a query along with its parent question and optional context. Your task is to select the correct action type based on the following rules:
### Available Action Types:
1. **Decomposition**
If the Query can be split into two logically related subqueries that together answer the parent query, respond as:
```json
{{"type": "decomposition", "subquery1": "...", "subquery2": "..."}}
```
2. **Entity Extraction**
If the Query lacks sufficient context to be decomposed, but contains key identifiable entities, extract them as:
```json
{{"type": "entity", "entity1": "...", "entity2": "..."}}
```
### Rules:
Ignore information not relevant to the query in the Query_context.
Ensure subquery1, subquery2, entity1, entity2 are all string values.
The output must be a single JSON object inside a markdown code block.
Do not provide any explanation or commentary outside the JSON.
### Examples:
**Example 1 (Decomposition):**
Query_context:  Doc 1: A Flame in My Heart is a 1987 French- Swiss drama film directed by Alain Tanner.
Parent_query of the Query: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Query: What is the birhday of the director of A Flame In My Heart?
Output: 
``` json
{{"type": "decomposition", "subquery1": "Who is Alain Tanner?", "subquery2": "What is the birhday of Alain Tanner?"}}
```
**Example 2 (Entity Extraction):**
Query_context: ...
Parent_query of the Query: Which film has the director born later, A Flame In My Heart or Butcher, Baker, Nightmare Maker?
Query: Who is the director of Butcher, Baker, Nightmare Maker?
Output: 
``` json 
{{"type": "entity", "entity1": "Butcher, Baker, Nightmare Maker", "entity2": "Director of Butcher, Baker, Nightmare Maker"}}
```
### Your Task:
Query_context: {context}
Parent_query of the Query: {parent_query}
Query: {query}
Output:

''')

    return user_prompt

def get_probtree_instruction():

    user_prompt = (
'''
Please generate a hierarchical question decomposition tree (HQDT) with json format for a given question. In this tree, the root node is the original complex question, and each non-root node is a sub-question of its parent. The leaf nodes are atomic questions that cannot be further decomposed.
Q: Jeremy Theobald and Christopher Nolan share what profession?
A: {"Jeremy Theobald and Christopher Nolan share what profession?": ["What is Jeremy Theobald's profession?", "What is Christopher Nolan's profession?"]}.
Q: How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?
A: {"How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?": ["In which South Korean television series Ryu Hye−young played Bo−ra?", "How many episodes were <1>?"]}.
Q: Vertical Limit stars which actor who also played astronaut Alan Shepard in "The Right Stuff"?
A: {"Vertical Limit stars which actor who also played astronaut Alan Shepard in \"The Right Stuff\"?": ["Vertical Limit stars which actor?", "Which actor played astronaut Alan Shepard in \"The Right Stuff\"?"]}.
Q: What was the 2014 population of the city where Lake Wales Medical Center is located?
A: {"What was the 2014 population of the city where Lake Wales Medical Center is located?": ["Which city was Lake Wales Medical Center located in?", "What was the 2014 population of <1>?"]}.
Q: Who was born first? Jan de Bont or Raoul Walsh?
A: {"Who was born first? Jan de Bont or Raoul Walsh?": ["When was Jan de Bont born?", "When was Raoul Walsh born?"]}.
Q: In what country was Lost Gravity manufactured?
A: {"In what country was Lost Gravity manufactured?": ["Which company was Lost Gravity manufactured?", "Which country is <1> in?"]}.
Q: Which of the following had a debut album entitled "We Have an Emergency": Hot Hot Heat or The Operation M.D.?
A: {"Which of the following had a debut album entitled \"We Have an Emergency\": Hot Hot Heat or The Operation M.D.?": ["What is the debut album of the band Hot Hot Heat?", "What is the debut album of the band The Operation M.D.?"]}.
Q: In which country did this Australian who was detained in Guantanamo Bay detention camp and published "Guantanamo: My Journey" receive para−military training?
A: {"In which country did this Australian who was detained in Guantanamo Bay detention camp and published \"Guantanamo: My Journey\" receive para−military training?": ["Which Australian was detained in Guantanamo Bay detention camp and published \"Guantanamo: My Journey\"?", "In which country did <1> receive para−military training?"]}.
Q: Does The Border Surrender or Unsane have more members?
A: {"Does The Border Surrender or Unsane have more members?": ["How many members does The Border Surrender have?", "How many members does Unsane have?"]}.
Q: James Paris Lee is best known for investing the Lee−Metford rifle and another rifle often referred to by what acronymn?
A: {"James Paris Lee is best known for investing the Lee−Metford rifle and another rifle often referred to by what acronymn?": ["James Paris Lee is best known for investing the Lee−Metford rifle and which other rifle?", "<1> is often referred to by what acronymn?"]}.
Q: What year did Edburga of Minster−in−Thanet's father die?
A: {"What year did Edburga of Minster−in−Thanet's father die?": ["Who is Edburga of Minster−in−Thanet's father?", "What year did <1> die?"]}.
Q: Were Lonny and Allure both founded in the 1990s?
A: {"Were Lonny and Allure both founded in the 1990s?": ["When was Lonny (magazine) founded?", "When was Allure founded?"]}.
Q: The actor that stars as Joe Proctor on the series "Power" also played a character on "Entourage" that has what last name?
A: {"The actor that stars as Joe Proctor on the series \"Power\" also played a character on \"Entourage\" that has what last name?": ["Which actor stars as Joe Proctor on the series \"Power\"?", "<1> played a character on \"Entourage\" that has what last name?"]}.
Q: How many awards did the "A Girl Like Me" singer win at the American Music Awards of 2012?
A: {"How many awards did the \"A Girl Like Me\" singer win at the American Music Awards of 2012?": ["Who is the singer of \"A Girl Like Me\"?", "How many awards did <1> win at the American Music Awards of 2012?"]}.
Q: Dadi Denis studied at a Maryland college whose name was changed in 1890 to honor what man?
A: {"Dadi Denis studied at a Maryland college whose name was changed in 1890 to honor what man?": ["Dadi Denis studied at which Maryland college?", "<1>'s name was changed in 1890 to honor what man?"]}.
Q: William Orman Beerman was born in a city in northeastern Kansas that is the county seat of what county?
A: {"William Orman Beerman was born in a city in northeastern Kansas that is the county seat of what county?": ["In which city in northeastern Kansas William Orman Beerman was born?", "<1> is the county seat of what county?"]}.

'''
    )


    return user_prompt

def get_probtree_cbprompt():
    user_prompt = (
'''
Please answer the question by thinking step-by-step.
Q: Jeremy Theobald and Christopher Nolan share what profession?
A: Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer. So the answer is: producer.
Q: How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?
A: The South Korean television series in which Ryu Hye−young played Bo−ra is Reply 1988. The number of episodes Reply 1988 has is 20. So the answer is: 20.
Q: Vertical Limit stars which actor who also played astronaut Alan Shepard in "The Right Stuff"?
A: The movie Vertical Limit starred actors including Chiris O'Donnell, Robin Tunney, Scott Glenn, etc. The actor who played astronaut Alan Shepard in "The Right Stuff" is Scott Glenn. So the actor who stars in Vertical Limit and played astronaut Alan Shepard in "The Right Stuff" is Scott Glenn. So the answer is: Scott Glenn.
Q: What was the 2014 population of the city where Lake Wales Medical Center is located?
A: Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140. So the answer is: 15,140.
Q: Who was born first? Jan de Bont or Raoul Walsh?
A: Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul Walsh was born the first. So the answer is: Raoul Walsh.
Q: In what country was Lost Gravity manufactured?
A: The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company. So the answer is: Germany.
Q: Which of the following had a debut album entitled "We Have an Emergency": Hot Hot Heat or The Operation M.D.?
A: The debut album of the band "Hot Hot Heat" was "Make Up the Breakdown". The debut album of the band "The Operation M.D." was "We Have an Emergency". So the answer is: The Operation M.D..
Q: Was Lonny (magazine) was founded in 2009?
A: Lonny (magazine) was founded in 2009. So the answer is: yes.
Q: In which country did this Australian who was detained in Guantanamo Bay detention camp and published "Guantanamo: My Journey" receive para−military training?
A: The Australian who was detained in Guantanamo Bay detention camp and published "Guantanamo: My Journey" is David Hicks. David Hicks received his para−military training in Afghanistan. So the answer is: Afghanistan.
Q: Does The Border Surrender or Unsane have more members?
A: The Border Surrender band has following members: Keith Austin, Simon Shields, Johnny Manning and Mark Austin. That is, it has 4 members. Unsane has following members: Chris Spencer, Cooper, and Jon Syverson. That is, it has 3 members. Thus, The Border Surrender has more members. So the answer is: The Border Surrender.
Q: James Paris Lee is best known for investing the Lee−Metford rifle and another rifle often referred to by what acronymn?
A: James Paris Lee is best known for investing the Lee−Metford rifle and Lee–Enfield series of rifles. Lee–Enfield is often referred to by the acronym of SMLE. So the answer is: SMLE.
Q: Was Lonny (magazine) was founded in 2008?
A: Lonny (magazine) was founded in 2009. So the answer is: no.
Q: What year did Edburga of Minster−in−Thanet's father die?
A: The father of Edburga of Minster−in−Thanet is King Centwine. Centwine died after 685. So the answer is: after 685.
Q: Were Lonny and Allure both founded in the 1990s?
A: Lonny (magazine) was founded in 2009. Allure (magazine) was founded in 1991. Thus, of the two, only Allure was founded in 1990s. So the answer is: no.
Q: The actor that stars as Joe Proctor on the series "Power" also played a character on "Entourage" that has what last name?
A: The actor that stars as Joe Proctor on the series "Power" is Jerry Ferrara. Jerry Ferrara also played a character on Entourage named Turtle Assante. Turtle Assante's last name is Assante. So the answer is: Assante.
Q: When was Jan de Bont born?
A: Jan de Bont was born on 22 October 1943. So the answer is: 22 October 1943.
Q: Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?
A: Nobody Loves You was written by John Lennon and released on the album Walls and Bridges. The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. So the answer is: Walls and Bridges.
Q: How many awards did the "A Girl Like Me" singer win at the American Music Awards of 2012?
A: The singer of "A Girl Like Me" is Rihanna. In the American Music Awards of 2012, Rihana won one award. So the answer is: one.
Q: Are both Bruce Chatwin and O. Henry writers？
A: Bruce Chatwin was an English travel writer, novelist, and journalist. O. Henry was an American writer. So both Bruce Chatwin and O. Henry are writers. So the answer is: yes.
Q: Which city is Lake Wales Medical Center located?
A: Lake Wales Medical Center is located in the city of Polk County, Florida. So the answer is: Polk County, Florida.
Q: Dadi Denis studied at a Maryland college whose name was changed in 1890 to honor what man?
A: Dadi Denis studied at the Maryland college Morgan State University. In 1890, the university's name was changed to honor Reverend Lyttleton Morgan. So the answer is: Reverend Lyttleton Morgan.
Q: William Orman Beerman was born in a city in northeastern Kansas that is the county seat of what county?
A: William Orman Beerman was born in Manhattan, Kansas. Manhattan, Kansas is the county seat of Riley County. So the answer is: Riley County.
''')
    
    return user_prompt

def get_probtree_obsinglehopprompt():
    user_prompt = (
'''
Given a question and the relevant Wikipedia text, answer the question and explain why. If you are unsure, answer Unknown.

#1 Wikipedia Title: 2014 Liqui Moly Bathurst 12 Hour
Text: The 2014 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars and Group 3E Series Production Cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 9 February 2014, was the twelfth running of the Bathurst 12 Hour.
#2 Wikipedia Title: 2015 Liqui Moly Bathurst 12 Hour
Text: The 2015 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars and Group 3E Series Production Cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 8 February 2015, was the thirteenth running of the Bathurst 12 Hour.
#3 Wikipedia Title: 2013 Liqui Moly Bathurst 12 Hour
Text: The 2013 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars, Group 3E Series Production Cars and Dubai 24 Hour cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 10 February 2013, was the eleventh running of the Bathurst 12 Hour. The race also incorporated the opening round of the 2013 Australian GT Championship. The Australian GT Championship was to compete as the first hour only and cars were permitted to enter for only that hour or to cross-enter for both the first hour and continue for the endurance race.
Q: Which track was the 2013 Liqui Moly Bathurst 12 Hour was staged?
A: The 2013 Liqui Moly Bathurst 12 Hour was staged at the Mount Panorama Circuit. So the answer is: Mount Panorama Circuit.

#1 Wikipedia Title: So Long, See You Tomorrow (album)
Text: So Long, See You Tomorrow is the fourth album by the London indie rock band Bombay Bicycle Club, released on 3 February 2014. The album is named after the novel of the same name by William Maxwell.
#2 Wikipedia Title: Hallelujah I Love Her So
Text: ``Hallelujah I Love Her So ''Single by Ray Charles from the album Ray Charles (or, Hallelujah I Love Her So) B - side`` What Would I Do Without You'' Released 1956 Format 7 ''45rpm Recorded 1956 Genre soul rhythm and blues Length 2: 35 Label Atlantic Songwriter (s) Ray Charles Producer (s) Jerry Wexler Ray Charles singles chronology ``A Fool for You'' (1955)`` Hallelujah I Love Her So ''(1956) ``Mary Ann'' (1956)`` A Fool for You ''(1955) ``Hallelujah I Love Her So'' (1956)`` Mary Ann ''(1956)
#3 Wikipedia Title: The First Time Ever I Saw Your Face
Text: ``The First Time Ever I Saw Your Face ''Single by Roberta Flack from the album First Take Released March 7, 1972 (1972 - 03 - 07) Recorded 1969 Genre Soul vocal jazz Length 5: 22 4: 15 (1972 radio edit) Label Atlantic 2864 Songwriter (s) Ewan MacColl Producer (s) Joel Dorn Roberta Flack singles chronology`` Will You Still Love Me Tomorrow'' (1972) ``The First Time Ever I Saw Your Face ''(1972)`` Where Is the Love'' (1972) ``Will You Still Love Me Tomorrow ''(1972)`` The First Time Ever I Saw Your Face'' (1972) ``Where Is the Love ''(1972)
Q: Is the performer of So Long, See You Tomorrow Bombay Bicycle Club?
A: The performer of So Long, See You Tomorrow is Bombay Bicycle Club. So the answer is: yes.

#1 Wikipedia Title: Oberoi family
Text: The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.
#2 Wikipedia Title: The Oberoi Group
Text: The Oberoi Group is a hotel company with its head office in Delhi. Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands.
#3 Wikipedia Title: Mohan Singh Oberoi
Text: Rai Bahadur Mohan Singh Oberoi (15 August 1898 – 3 May 2002) was an Indian hotelier, the founder and chairman of Oberoi Hotels & Resorts, India's second-largest hotel company, with 35 hotels in India, Sri Lanka, Nepal, Egypt, Australia and Hungary.
Q: The Oberoi family is part of which hotel company?
A: The Oberoi family is part of the hotel company The Oberoi Group. So the answer is: The Oberoi Group.
''')
    return user_prompt

def get_probtree_obmultihopprompt():

    user_prompt = (
'''
Please answer the question and explain why. Output no more than 5 words after "So the answer is".

#1 Wikipedia Title: First (magazine)
Text: FiRST is a Singaporean movie magazine formerly published monthly, now running as a weekly newspaper insert.
#2 Wikipedia Title: Arthur's Magazine
Text: Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into "Godey's Lady's Book".
#3 Wikipedia Title: First for Women
Text: First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011 the circulation of the magazine was 1,310,696 copies.
#4 Wikipedia Title: First Eleven (magazine)
Text: First Eleven is a British specialist magazine for parents of children at independent schools.
#5 Wikipedia Title: Earth First! (magazine)
Text: Earth First!, the radical environmental journal, is the official publication of the Earth First! movement. First published as a newsletter in 1980, it has existed alongside the movement as a way to spread commonly held beliefs in "Earth First!" culture, such as biocentrism, deep ecology, and direct action. The magazine is also commonly known as the "Earth First! Journal".
Q: Which magazine was started first Arthur's Magazine or First for Women?
A: Arthur's Magazine was started in 1844. First for Women was started in 1989. So Arthur's Magazine was started first. So the answer is: Arthur's Magazine.

#1 Wikipedia Title: The Oberoi Group
Text: The Oberoi Group is a hotel company with its head office in Delhi. Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands.
#2 Wikipedia Title: The Body Has a Head
Text: The Body Has a Head is an album by King Missile frontman John S. Hall, released exclusively in Germany in 1996. Though billed as a Hall "solo album," the collection features considerable input from multi-instrumentalists Sasha Forte, Bradford Reed, and Jane Scarpantoni, all of whom would become members of the next incarnation of King Missile ("King Missile III") and contribute to that group's "debut" album, 1998's "Failure."
#3 Wikipedia Title: Oberoi family
Text: The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.
#4 Wikipedia Title: Has-a
Text: In database design, object-oriented programming and design (see object oriented program architecture), has-a (has_a or has a) is a composition relationship where one object (often called the constituted object, or part/constituent/member object) "belongs to" (is part or member of) another object (called the composite type), and behaves according to the rules of ownership. In simple words, has-a relationship in an object is called a member field of an object. Multiple has-a relationships will combine to form a possessive hierarchy.
#5 Wikipedia Title: Oberoi Realty
Text: Oberoi Realty is a real estate developer based in Mumbai, Maharashtra. It is led by Mr. Vikas Oberoi, CMD. The company has developed over 39 projects at locations across Mumbai. Its main interest is in Residential, Office Space, Retail, Hospitality and Social Infrastructure properties in Mumbai.
Q: The Oberoi family is part of a hotel company that has a head office in what city?
A: The Oberoi family is part of a hotel company The Oberoi Group. The Oberoi Group has a head office in Delhi. So the answer is: Delhi.

#1 Wikipedia Title: 2014 Liqui Moly Bathurst 12 Hour
Text: The 2014 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars and Group 3E Series Production Cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 9 February 2014, was the twelfth running of the Bathurst 12 Hour.
#2 Wikipedia Title: 2015 Liqui Moly Bathurst 12 Hour
Text: The 2015 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars and Group 3E Series Production Cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 8 February 2015, was the thirteenth running of the Bathurst 12 Hour.
#3 Wikipedia Title: 2013 Liqui Moly Bathurst 12 Hour
Text: The 2013 Liqui Moly Bathurst 12 Hour was an endurance race for a variety of GT and touring car classes, including: GT3 cars, GT4 cars, Group 3E Series Production Cars and Dubai 24 Hour cars. The event, which was staged at the Mount Panorama Circuit, near Bathurst, in New South Wales, Australia on 10 February 2013, was the eleventh running of the Bathurst 12 Hour. The race also incorporated the opening round of the 2013 Australian GT Championship. The Australian GT Championship was to compete as the first hour only and cars were permitted to enter for only that hour or to cross-enter for both the first hour and continue for the endurance race.
#4 Wikipedia Title: Mount Panorama Circuit
Text: Mount Panorama Circuit is a motor racing track located in Bathurst, New South Wales, Australia. It is situated on a hill with the dual official names of Mount Panorama and Wahluu and is best known as the home of the Bathurst 1000 motor race held each October, and the Bathurst 12 Hour event held each February. The 6.213 km long track is technically a street circuit, and is a public road, with normal speed restrictions, when no racing events are being run, and there are many residences which can only be accessed from the circuit.
#5 Wikipedia Title: List of Mount Panorama races
Text: This is a list of significant car races that have been held at the Mount Panorama Circuit near Bathurst, New South Wales, Australia. As Australia's most famous motor racing circuit, Mount Panorama has had a significant influence on the history and industry of Australian motor racing.
Q: What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?
A: The 2013 Liqui Moly Bathurst 12 Hour was staged at the Mount Panorama Circuit. Mount Panorama Circuit is 6.213 km long. So the answer is: 6.213 km long.
''')
    return user_prompt

def get_probtree_aggregate_prompt():
    user_prompt = (
'''
Given a qeustion and a context, answer the question and explain why.

#
Context:
Which famous fashion show Stella Maxwell has been a model for? Victoria's Secret.
Since when Victoria's Secret? 1977.

Question:
Stella Maxwell has been a model for a famous fashion shown since when?

Answer:
Stella Maxwell has been a model for a famous fashion shown, Victoria's Secret since 2015. So the answer is: since 2015.
#
Context:
Who is the American retired professional basketball player who is current president of basketball operations for the Los Angeles Lakers? Devean George.
William Novac co-wrote the memoir of Devean George? no.

Question:
William Novac co-wrote the memoir of what American retired professional basketball player who is current president of basketball operations for the Los Angeles Lakers?

Answer:
William Novac co-wrote the memoir of Magic Johnson, an American retired professional basketball player who is current president of basketball operations for the Los Angeles Lakers. So the answer is: Magic Johnson.
#
Context:
Which athlete rode 400 miles across his country to bring attention to the plight of the disabled in the country? Emmanuel Ofosu Yeboah.
What is the title of the documentary narrated by Oprah Winfrey about Emmanuel Ofosu Yeboah? Emmanuel's Gift.

Question:
Oprah Winfrey narrated a documentary about this athlete who rode 400 miles across his country to bring attention to the plight of the disabled in the country?

Answer:
Oprah Winfrey narrated a documentary about the athelete Emmanuel Ofosu Yeboah, who rode 400 miles across his country to bring attention to the plight of the disabled in the country. So the answer is: Emmanuel Ofosu Yeboah.
#
''')
    return user_prompt