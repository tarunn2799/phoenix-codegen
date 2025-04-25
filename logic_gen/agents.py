import litellm
from litellm import completion

class Agent:
    def __init__(self, sys_prompt: str, feedback_prompt: str = None, model: str = "gpt-4"):
        """
        Initialize an Agent with a system prompt and optional feedback prompt.
        `model` specifies the LLM model to use (default is gpt-4).
        """
        self.sys_prompt = sys_prompt
        self.feedback_prompt = feedback_prompt
        self.model = model
        # Start conversation with the system prompt
        self.messages = [{"role": "system", "content": self.sys_prompt}]
    
    def llm_call(self, prompt: str, feedback: bool = False):
        """
        Send a prompt to the LLM. If feedback=False, it uses the prompt directly.
        If feedback=True, it formats the prompt using the agent's feedback_prompt template.
        Returns the full message history (including the newly added assistant response).
        """
        if feedback:
            if not self.feedback_prompt:
                raise ValueError("Feedback prompt is not set for this agent.")
            # Format the feedback into the user message
            user_content = self.feedback_prompt.format(feedback=prompt)
        else:
            user_content = prompt
        # Append the user message and get LLM completion
        self.messages.append({"role": "user", "content": user_content})
        response = completion(model=self.model, temperature=0.1, messages=self.messages)
        # Extract response text and append to conversation history
        if hasattr(response, "choices"):
            response_text = response.choices[0].message['content']
        else:
            # In case the completion return is a dict-like object
            response_text = response['choices'][0]['message']['content']
        self.messages.append({"role": "assistant", "content": response_text})
        return self.messages
    
    def reset(self):
        """Reset the conversation, preserving only the initial system prompt."""
        self.messages = [{"role": "system", "content": self.sys_prompt}]