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


def get_native_instruction():
    user_prompt = (
        '''
            Answer the following question:
            Question: {question}
            IMPORTANT: You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.
            Answer:
        '''
    )
    return user_prompt

def get_rag_instruction():
    user_prompt = (
        '''
            Answer the following question:
            You can use the following documents to help you answer the question.
            Documents: {context}
            IMPORTANT: You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.
            Question: {question}
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


def get_final_answer_llama3_8b():
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