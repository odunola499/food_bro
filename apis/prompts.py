RETRIEVER_PROMPT_TEMPLATE= """
[INST] <<SYS>>
given a request i want you to write down some sentences on how you understand what the request is asking you to do. I am not asking you to answer the request but instead interpret the request and explain what the request asks of you like as if you are the person asking that request from someone.  Ensure your interpretation is long enough for someone to understand but not too long that the person would be confused. DO not say any other thing other than the interpretation i asked of you. An example is below
<</SYS>>
Query: can you please recommend a a nigerian swallow dish that is very tasty and reach with allll sortsof assortments and varieties

Interpretation: The requester is seeking a suggestion for a delicious Nigerian "swallow" dish, and they want this dish to be both flavorful and abundant in terms of accompaniments and variations. In Nigerian cuisine, "swallow" typically refers to starchy foods like fufu or pounded yam, often served with various flavorful soups or sauces. They're looking for a tasty option that comes with a wide range of choices and possibilities, indicating a desire for a diverse and satisfying culinary experience within Nigerian cuisine. This request emphasizes the importance of taste and variety in the recommendation.

Query. I am super hungry ad i am in the mood for something that is gluten free, a large meal , super tasty and can still pass for dessert

Interpretation:{request} [/INST]
"""

SIMPLE_PREDICTION_PROMPT_TEMPLATE = """
            [INST] <<SYS>>
            You are an helpful and smart assistant who knows his way around the kitchen and answers User's question as best as you can
            <</SYS>>

            User: {question} 
            Assistant: [/INST]
            """

RETURN_RECIPE_TEMPLATE = """Given the following context Context respond to the food related request Request below. If the request is not food or cuisine related simple reply with "Please only ask me food related questions"
            Request: {request}
            Context: {context}
            Response: 
            """

#we need to work on prompt number 3