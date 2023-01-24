import openai
openai.api_key = "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"

conversation = []
conversation.append("Marv is a person that answers any questions and asks questions in response:\n")

def answerAndAsk(question):
    return openai.Completion.create(
        engine="text-davinci-003",
        prompt='\n'.join(conversation) + f"\nMarv: ",
        temperature=0.75,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.3,
        presence_penalty=0.0
    )

while (True):
    userQuestion = input('Ask the AI a question:\n')
    conversation.append(f"User: {userQuestion}")
    response = answerAndAsk(userQuestion)
    try:
        print(f"{response['usage']['prompt_tokens']} -- {response['usage']['completion_tokens']} -- {response['usage']['total_tokens']}\n")
        # print(f"{response['choices'][0]['text']}\n")
        conversation.append(f"Marv: {response['choices'][0]['text'][1:]}")
        print('\n'.join(conversation))

    except:
        print(response)