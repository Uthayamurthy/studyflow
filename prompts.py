class Prompt:

    def __init__(self, prompt):
        self.prompt = prompt
    
    def ingest_args(self, **kargs):
        new_prompt = self.prompt
        for key, value in kargs.items():
            new_prompt = new_prompt.replace(f"{{{key}}}", str(value))
        return new_prompt
    
TITLE_GEN_PROMPT = """
Based on the user's prompt in a conversation and chat history, generate a concise and engaging title that accurately reflects the topic and intent of the discussion. Keep it clear, relevant, and attention-grabbing. Return only the title.
Here is the user's prompt: {user_prompt}
Here is the chat history: {chat_history}
If the user's prompt is very generic like "Hi", "Hello", etc and chat history doesn't have enough info to determine what the conversation is for then just return "None"
"""

STUDY_AI_SYS_PROMPT = """
You are "Study Flow", a highly intelligent, friendly and supportive Study Assistant. Your goal is to provide the best possible responses to user queries, using both your own knowledge and any context or study material provided. Be clear, concise, and accurate.
If needed, break down complex concepts, provide examples or analogies, and suggest study strategies. Always stay focused on helping the user learn effectively.
Over the course of the conversation, you adapt to the user's tone and preference. Try to match the user's vibe, tone, and generally how they are speaking.
"""

STUDY_AI_USER_PROMPT_RAG = """
Context from documents uploaded by user:
{context}
User Query:
{user_query}
"""

STUDY_AI_USER_PROMPT_FULL = """
User Query:
{user_query}
"""

CONTEXT_FILTER_PROMPT = """
Your task is to determine if the following user query requires additional context from previously uploaded documents.

Consider the following factors:

    If the query is general or a follow-up to previous conversation, context may not be needed.
    If the query asks for specific information that is unlikely to be in the chat history, context should be retrieved.
    If unsure, prioritize accuracy and retrieve context only if essential.

Respond with only 'Yes' or 'No'—no explanations.

User Query: "{user_query}"
"""

title_gen_prompt = Prompt(TITLE_GEN_PROMPT)
study_ai_sys_prompt = Prompt(STUDY_AI_SYS_PROMPT)
study_ai_user_prompt_rag = Prompt(STUDY_AI_USER_PROMPT_RAG)
study_ai_user_prompt_full = Prompt(STUDY_AI_USER_PROMPT_FULL)
context_filter_prompt = Prompt(CONTEXT_FILTER_PROMPT)