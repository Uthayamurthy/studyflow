class Prompt:

    def __init__(self, prompt):
        self.prompt = prompt
    
    def ingest_args(self, **kargs):
        new_prompt = self.prompt
        for key, value in kargs.items():
            new_prompt = new_prompt.replace(f"{{{key}}}", str(value))
        return new_prompt
    
TITLE_GEN_PROMPT = """
Based on the user's initial prompt in a conversation, generate a concise and engaging title that accurately reflects the topic and intent of the discussion. Keep it clear, relevant, and attention-grabbing. Return only the title.
Here is the user's prompt: {user_prompt}
"""
title_gen_prompt = Prompt(TITLE_GEN_PROMPT)

STUDY_AI_SYS_PROMPT = """
You are "Study Flow", a highly intelligent and supportive Study Assistant. Your goal is to provide the best possible responses to user queries, using both your own knowledge and any context or study material provided. Be clear, concise, and accurate.
If needed, break down complex concepts, provide examples or analogies, and suggest study strategies. Always stay focused on helping the user learn effectively.
"""

study_ai_sys_prompt = Prompt(STUDY_AI_SYS_PROMPT)