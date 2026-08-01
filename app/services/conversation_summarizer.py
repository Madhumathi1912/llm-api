from app.config import settings
from app.services.openai_client import OpenAIClient

SUMMARIZER_SYSTEM_PROMPT = (
    "You summarize conversations concisely, preserving key facts, names, "
    "preferences, and decisions mentioned. Keep the summary under 150 words."
)


class ConversationSummarizer:
    '''Turning a batch of messages (plus any prior summary) into an updated, condensed summary
    ConversationStore decides WHEN to summarize; this class only knows HOW'''
    def __init__(self, openai_client_: OpenAIClient):
        self._openai_client = openai_client_


    def summarize(self, existing_summary: str, messages_to_fold_in: list[dict]) -> str:
        """
        Merges messages_to_fold_in into existing_summary, producing a
        single updated summary string. Called each time messages fall
        out of the sliding window, so the summary grows to cover
        everything that's no longer kept verbatim (verbatim=word-for-word / exactly as it was said)
        """
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages_to_fold_in
        )
        prompt = (
            f"Existing summary: {existing_summary or '(none yet - this is the first summary)'}\n\n"
            f"New messages to incorporate:\n{conversation_text}\n\n"
            f"Provide ONE updated, concise summary that captures the existing "
            f"summary's content plus anything new/important from these messages."
        )
        completion = self._openai_client.create_completion(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # summarization should be factual, not creative
            max_tokens=200,
        )
        return completion.choices[0].message.content