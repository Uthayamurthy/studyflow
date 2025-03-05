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