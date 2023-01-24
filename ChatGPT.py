from revChatGPT.ChatGPT import Chatbot

chatbot = Chatbot({
  "session_token": "cfa8b5ba94e3f0e7fa0ea3d5f5f3a7010ff690b0a8482c54df25111c2c0edd21%7C5c0ded0da9a2eb288f0245e2abd64ce9caa6d01551af6c61278a2563d46f9dfe"
}, conversation_id=None, parent_id=None) # You can start a custom conversation

response = chatbot.ask("What is your favorite color?", conversation_id=None, parent_id=None) # You can specify custom conversation and parent ids. Otherwise it uses the saved conversation (yes. conversations are automatically saved)

print(response)
# {
#   "message": message,
#   "conversation_id": self.conversation_id,
#   "parent_id": self.parent_id,
# }