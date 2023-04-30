#!/usr/bin/env python3
"""
import os
import openai

os.environ["OPENAI_API_KEY"] = ""

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(model="text-davinci-003", prompt="Say this is a test", temperature=0, max_tokens=7)
"""
import sys, threading, time
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QComboBox, QLabel, QCheckBox
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QSize
import openai
import whisper
import speech_recognition as sr
import numpy as np
import torch
import numpy as np
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from translate import Translator


print("Loading whisper engine...")
audio_model = whisper.load_model("small")

print("Starting speech recognition engine...")
r = sr.Recognizer()
r.recognize_whisper_api
r.energy_threshold = 300
r.pause_threshold = 0.8
r.dynamic_energy_threshold = False




class SendToBot(threading.Thread):
    def __init__(self, messageHistory):
        threading.Thread.__init__(self)
        self.messageHistory = messageHistory
        self.result = None

    def run(self):

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messageHistory
        )['choices'][0]['message']['content'].strip()

        self.result = response


class TextToSpeech(threading.Thread):
    def __init__(self, message, lang):
        threading.Thread.__init__(self)
        self.message = message
        self.lang = lang

    def run(self):
        tts = gTTS(self.message, lang=self.lang)
        mp3_file = BytesIO()
        tts.write_to_fp(mp3_file)
        mp3_file.seek(0)
        audio_segment = AudioSegment.from_file(mp3_file, format="mp3")
        play(audio_segment)


defaultPrompts = {}
defaultPrompts['english'] = "For the purpose of helping someone learn a new language, give human-like responses to all questions and ask questions in response. Act with a backstory, such as favorite foods, location of living, a human name, and more."

difficultyPrompts = {}
difficultyPrompts[('english','beginner')] = "Please respond with easy and simple responses using basic vocabulary because the user is still a beginner. Also try to only ask one question at a time."
difficultyPrompts[('english','intermediate')] = "Please respond with easy and simple responses using mostly basic vocabulary because the user is still only at an indermediate level. Sometimes ask more than one question in response."
difficultyPrompts[('english','advanced')] = "The user is an advanced level user of the language, but they are still learning, so try to make it easy enough to understand and learn from your responses."


languageAbbreviations = {'english':'en', 'spanish':'es', 'japanese':'ja', 'german':'de'}

chatHistory = []
selectedLanguage = 'spanish'
knownLanguage = 'english'
languageSkill = 'beginner'
ttsState = False

def generateNewChat():
    global chatHistory
    chatHistory = []
    chatHistory.append({"role": "system", "content": defaultPrompts[knownLanguage]+"\n"+difficultyPrompts[(knownLanguage,languageSkill)]})
generateNewChat()


is_Sent_Message = False

class ChatbotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setMinimumSize(1235, 705)
        self.setWindowTitle("ChatGPT Language Learning Program")

        self.chat_area = QTextEdit(self)
        self.chat_area.setGeometry(20,20,970,600)
        self.chat_area.setReadOnly(True)

        self.input_field = QLineEdit(self)
        self.input_field.setGeometry(20, 640, 840, 20)

        self.translational_field = QLabel(self)
        self.translational_field.setGeometry(20, 665, 840, 20)

        self.send_button = QPushButton("", self)
        self.send_button.setGeometry(885,640,45,45)
        size = QSize(21, 21)
        self.send_button.setIcon(QIcon('sendIcon.png'))
        self.send_button.setIconSize(size)

        self.mic_button = QPushButton("", self)
        self.mic_button.setGeometry(945,640,45,45)
        self.mic_button.setIcon(QIcon('micIcon.png'))
        self.mic_button.setIconSize(size)

        self.toLearnLabel = QLabel(self)
        self.toLearnLabel.setText("Language to learn:")
        self.toLearnLabel.setGeometry(1010, 20, 200, 20)
        self.language_choice = QComboBox(self)
        self.language_choice.addItem("English")
        self.language_choice.addItem("Spanish (Español)")
        self.language_choice.addItem("German (Deutsch)")
        self.language_choice.addItem("Japanese (日本語)")
        self.language_choice.setCurrentText("Spanish (Español)")
        self.language_choice.setGeometry(1010, 40, 200, 30)
        self.language_choice.currentIndexChanged.connect(self.on_language_changed)

        self.knownLabel = QLabel(self)
        self.knownLabel.setText("Known starting language:")
        self.knownLabel.setGeometry(1010, 80, 200, 20)
        self.knownLanguage_choice = QComboBox(self)
        self.knownLanguage_choice.addItem("English")
        self.knownLanguage_choice.addItem("Spanish (Español)")
        self.knownLanguage_choice.setGeometry(1010, 100, 200, 30)
        self.knownLanguage_choice.currentIndexChanged.connect(self.on_known_language_changed)

        self.skillLabel = QLabel(self)
        self.skillLabel.setText("Skill level:")
        self.skillLabel.setGeometry(1010, 140, 200, 20)
        self.skill_choice = QComboBox(self)
        self.skill_choice.addItem("Beginner")
        self.skill_choice.addItem("Intermediate")
        self.skill_choice.addItem("Advanced")
        self.skill_choice.setGeometry(1010, 160, 200, 30)
        self.skill_choice.currentIndexChanged.connect(self.on_skill_changed)

        self.ttsLabel = QLabel(self)
        self.ttsLabel.setText("Text-to-speech:")
        self.ttsLabel.setGeometry(1010, 200, 200, 20)
        self.ttsToggle = QCheckBox(self)
        self.ttsToggle.setGeometry(1095, 202, 20, 20)
        self.ttsToggle.setChecked(False)
        self.ttsToggle.stateChanged.connect(self.on_tts_toggled)


        self.show()

        self.send_button.clicked.connect(self.send_message)
        self.mic_button.clicked.connect(self.micPressed)





    def on_language_changed(self):
        self.chat_area.clear()
        global selectedLanguage, knownLanguage, languageSkill
        selectedLanguage = self.language_choice.currentText().split()[0].lower()
        generateNewChat()
        print(f"Translation: {knownLanguage} --> {selectedLanguage} [{languageSkill}]")


    def on_known_language_changed(self):
        self.chat_area.clear()
        global selectedLanguage, knownLanguage, languageSkill
        knownLanguage = self.knownLanguage_choice.currentText().split()[0].lower()
        if (knownLanguage not in defaultPrompts) or ((knownLanguage,languageSkill) not in difficultyPrompts):
            translator = Translator(from_lang='en', to_lang=languageAbbreviations[knownLanguage])
            defaultPrompts[knownLanguage] = translator.translate(defaultPrompts['english'])
            difficultyPrompts[(knownLanguage,'beginner')] = translator.translate(difficultyPrompts[('english','beginner')])
            difficultyPrompts[(knownLanguage,'intermediate')] = translator.translate(difficultyPrompts[('english','intermediate')])
            difficultyPrompts[(knownLanguage,'advanced')] = translator.translate(difficultyPrompts[('english','advanced')])
        generateNewChat()
        print(f"Translation: {knownLanguage} --> {selectedLanguage} [{languageSkill}]")


    def on_skill_changed(self):
        self.chat_area.clear()
        global selectedLanguage, knownLanguage, languageSkill
        languageSkill = self.skill_choice.currentText().split()[0].lower()
        generateNewChat()
        print(f"Translation: {knownLanguage} --> {selectedLanguage} [{languageSkill}]")

    def on_tts_toggled(self):
        global ttsState
        ttsState = not ttsState
        print(f"TTS: {ttsState}")





    def send_message(self):
        msg = self.input_field.text().strip()

        if not msg or len(msg) < 1:
            return
        
        global chatHistory
        chatHistory.append({"role": "user", "content": msg})
        chatBotThread = SendToBot(chatHistory)
        chatBotThread.start()
        self.add_to_ui('Me', msg)
        self.input_field.clear()
        chatBotThread.join()

        global ttsState, selectedLanguage
        if (ttsState):
            ttsThread = TextToSpeech(chatBotThread.result, languageAbbreviations[selectedLanguage])
            ttsThread.start()
            
        
        time.sleep(7) # wait for message before adding it
        chatHistory.append({"role": "assistant", "content": chatBotThread.result})
        self.add_to_ui('Bot', chatBotThread.result)

        
        chatBotThread.join()
    




    def micPressed(self):
        print("\nListening...")
        self.input_field.setText("Listening...")
        self.input_field.setReadOnly(True)
        self.mic_button.setDisabled(True)
        with sr.Microphone(sample_rate=16000) as source:
            audio = r.listen(source)
            torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
            global selectedLanguage
            print(f"Transcribing using {selectedLanguage}")
            output_text = audio_model.transcribe(torch_audio, fp16=False, language=selectedLanguage)["text"].strip()
            translator = Translator(from_lang=languageAbbreviations[selectedLanguage], to_lang=languageAbbreviations[knownLanguage])
            backToKnown = translator.translate(output_text)
            self.translational_field.setText(backToKnown)
            self.input_field.setReadOnly(False)
            self.input_field.setText(output_text)
            self.mic_button.setEnabled(True)
            print(output_text)






    def add_to_ui(self, id, msg):
        global selectedLanguage, knownLanguage
        translator = Translator(from_lang=languageAbbreviations[selectedLanguage], to_lang=languageAbbreviations[knownLanguage])
        backToKnown = translator.translate(msg)
        self.chat_area.append(f'{id}: {msg}')
        self.chat_area.append(f'        {backToKnown}\n')


app = QApplication(sys.argv)
mainWindow = ChatbotWindow()
sys.exit(app.exec())
