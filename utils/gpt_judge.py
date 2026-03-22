from utils.system_instructions import SYSTEM_PROMPTS

def calculate_relevance(prompt, response, client, model = "gpt-4.1-mini"):
    system_prompt = SYSTEM_PROMPTS["relevance_scoring"]

    prompt_and_response = """Input Query: {}\nChatbot Response: {}"""
    prompt_and_response = prompt_and_response.format(f"{prompt}", f"{response}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_and_response}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        response = int(response.choices[0].message.content.strip())
        if response == 0:
            return -1
        elif response == 1:
            return 0
        elif response == 2:
            return 1
        else:
            # Raise an error for unexpected output
            raise ValueError(f"Unexpected model output: {response}")

    except Exception as e:
        print("Error in GPT relevance evaluation:", e)
        return -2