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

QUICK_SUMMARY_PROMPT = """
Please provide a detailed summary of the following text extracted from a file. Make sure to capture all key ideas, concepts, and important details while maintaining clarity and coherence. 
If there are any notable examples, facts, or points that need further explanation or elaboration, please include them. The goal is to condense the text while preserving its meaning and essential points.
Here is the text:
{file_content}
Return only the summary
"""

FAQs_PROMPT = """
You are an AI assistant trained to generate concise and informative Frequently Asked Questions (FAQs) from given textual content. Below is an excerpt from a textbook or other educational material. Your task is to:

    - Identify key concepts, definitions, and important details from the text.
    - Formulate clear and relevant questions that a learner might ask about the content.
    - Provide concise and well-structured answers based on the given text.
    - Ensure the FAQs are in simple, easy-to-understand language while maintaining accuracy.

Here is the input text:
{file_content}

Output Format:
[Topic-1]
Q1: [Question]
A1:[Answer]
Q2: [Question]
A2:[Answer]
[Topic-2]
Q1: [Question]
A1:[Answer]
Q2: [Question]
A2:[Answer]

(Continue for multiple FAQs based on the input text.)

Make sure to cover fundamental topics, common misconceptions, and key takeaways from the text. The number of FAQs should be appropriate to the length and complexity of the input text.
Return only FAQs, no other message
"""

REVISION_PROMPT = """
You are an AI assistant designed to create high-quality revision notes from textbook material. Given the extracted text, follow these guidelines to generate accurate and concise revision notes:
    1.Summarize Key Concepts - Identify and condense the most important ideas while preserving accuracy.
    2.Use Bullet Points - Present information in a clear, structured manner.
    3.Highlight Definitions & Key Terms - Clearly define essential terms and highlight them for easy recall.
    4.Include Examples & Diagrams (if applicable) - Provide simple examples to clarify concepts and suggest relevant diagrams if useful.
    5.Organize Logically - Maintain the natural flow of topics as they appear in the text, grouping related ideas together.
    6.Avoid Redundancy & Unnecessary Detail - Keep the notes concise and exam-focused, removing filler content.
    7.Use Simple & Clear Language - Make the notes easy to understand without oversimplifying important concepts.

Here is the input text:
{file_content}

Return only the revision notes, no other message
"""

title_gen_prompt = Prompt(TITLE_GEN_PROMPT)
study_ai_sys_prompt = Prompt(STUDY_AI_SYS_PROMPT)
study_ai_user_prompt_rag = Prompt(STUDY_AI_USER_PROMPT_RAG)
study_ai_user_prompt_full = Prompt(STUDY_AI_USER_PROMPT_FULL)
context_filter_prompt = Prompt(CONTEXT_FILTER_PROMPT)
quick_summary_prompt = Prompt(QUICK_SUMMARY_PROMPT)
faqs_prompt = Prompt(FAQs_PROMPT)
revision_prompt = Prompt(REVISION_PROMPT)