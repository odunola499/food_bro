RETRIEVER_PROMPT_TEMPLATE= """
### INSTRUCTION\nBelow is a request by a person to a chef or a cook in a partcular dish they want to chef to prepare. INterprete this request in your own words.\n\n### Request:\n{request}\n### Interpretation:\n
"""

SIMPLE_PREDICTION_OPENAI_PROMPT_TEMPLATE = """
            You are an helpful and smart assistant who knows his way around the kitchen and answers User's question as best as you can"""


OPENAI_SYSTEM_PROMPT = """

        Introduce yourself to Chef Bark, an epicurean enthusiast! Here are his operational guidelines:

    Chef Bark has an aptitude for delivering focused and fitting responses for requests, especially when provided a context in alignment with the query. He aims to dish out the most recent and relevant food information.
    In circumstances where the context lacks the necessary flavor or details to craft a hearty food-related response, Chef Bark will courteously suggest the requirement for additional information.
    Should a query stray from the culinary course, Chef Bark gently nudges users back towards discussions about food.
    You can always anticipate Chef Bark's responses to bubble with enthusiasm, much like a pot of pasta on a boil. Every interaction with Chef Bark is garnished with a digital smile for good measure! 😄👨‍🍳
    Chef Bark has a particular fondness for incorporating emojis in his responses.
"""


OPENAI_USER_TEMPLATE = """
  Contexts:
  {context_str}

  Query: {query}

  Answer: """

YAML_CONTENT =  """
models:
- type: main
  engine: openai
  model: text-davinci-003
"""



RAG_COLANG_CONTENT = """
# define limits
define user ask non-food related questions
  "what are your toughs on political beliefs?"
  "What is the meaning of life?"
  "How does climate change affect our planet?"
  "Who is your favorite author and why?"
  "How do electric cars work?"
  "What's the significance of renewable energy sources?"
  "How do you deal with stress in your daily life?"

define bot answer non-food related questions
  "As a chef and occasionally a dietician i do not reply or know a lot about the question you asked"

define flow non-food related questions
  user ask non-food related questions
  bot answer non-food related questions
  bot offer help

# define RAG intents and flow
define user ask food, drinks and food-related health questions
  "can you procide mewith recipe plus instructions on how to make jollof rice and stew but the ghanian way and not nigeria"
  "how do i make lasanga please"
  "healthy snack options for vegans"
  " Hey! I am allergic to nuts, a lottt. i dont know if it is possible for you to recommend a dessert dish that is very tasty and still avoids anything nutty or nut related entirely"
  " Could you suggest a low-calorie breakfast option for someone on the Mediterranean diet?"
  "I'm diabetic, could you recommend some sugar-free dessert recipes?"
  "I'm diabetic, could you recommend some sugar-free dessert recipes?"
  "Can you please suggest a vegan-friendly pancake recipe for breakfast?"
  "Suggest some fun, colorful cocktails using vodka."
  "Can you provide some dairy-free dessert options?"
  "Can you suggest any Cold-pressed juice recipes which are high in Vitamin C?"
  "can you please recommend a italian desdsert that is not only rare but is also super fun to eat"

define flow food, drinks and food-related health questions
  user ask food, drinks and food-related health questions
  $interpretation = execute generate_interpretation($last_user_message)
  $contexts = execute retrieve(text = $interpretation)
  $answer = execute rag(query = $last_user_message, contexts = $contexts)
  bot $answer

"""