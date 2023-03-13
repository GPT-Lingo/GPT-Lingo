from revChatGPT.ChatGPT import Chatbot
import whisper
import speech_recognition as sr
import numpy as np
import torch
import numpy as np
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play


print("Starting ChatGPT wrapper...")
chatbot = Chatbot({
  "session_token": "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..gRqb4B53vfMGgA8a.6RX-XJiwv4TU-uxe70RljzsqKueCgD9aXlH3hAm7eb1mtswp_SCA-OY3ErsDGGTwHfEcOyod4U55nsoGlXr1EiT1DrgZ7AcEnvFsqxprhDQTP2I0W1CVXnkuFRiZaXFC2cf0W3oZAJl0EelPcKHOdAtvEl1s22gdwFaw9TPlY2ArgCOijxWimGabsecPeXHGPUK7GKd431oJhiHex0ZC2DAqBgwCjlKBupkV70KGkteMv1bG4UwP-WpQBHjIUMgXGkAkcc93tMkykQK3DWDKXNH8tLXL-WNOM6qpIAO6s_rkYdrnjM5-WvDMvH1iCKKPtJ7NnKsBNtFb0VD-6alR9bL9Gk_Y2vbqeudZNApNI6QzQ7IKkml1BzhL6fFgLGs2hubfzgWV13vbuBPscV_YBlkJ1Q8GPWZt7gHi9iK968flphiI-kLahweLam47mnwmYxfWhP6C5uXE1HVow9hMb3MZRcmpRQAPbtjMfUJ7VNY7JOPbxES_6vgRLFC12qMIfvQDZfx59dboIjdxHlv8r2R5xCiNzq0aiFlD2Ng0bQc042lFiSkXK3aHqjVFhOWfg5iuNj6Uq6T4mrZLUILlxQjlEaWjzvojjKyIs8GnaMptWGf0q9lZeVKLMmI0Mxlgvr1l3w5FxObXWbyPBskHijKgnU3-Qrn-e7lkQajJ8OO1GpJnDX60XL_D52x37SxRLWJ6sx7kgfWLCZghlSrnMAL30aIZCvr2UZBG0tJnqEaPjvbU-ZBGJ4r8eOKGbJrTL5mKIbnPAXp-tmUS6CjyX52l-G_kZ4tdFHO7aDP0mg41OVVJiEPKnUr0S00LGL_ML_XALsnSiqhrmm4jqT40ifDuUzO-Sr1Md8YzEPdc55QJYcb7Z8Sx-ZNJV5KYJcvMB9WUW8j6tSUx12iDgcc9VS4K_abmB2HO2Zx4gUll72EO4mZ0WumJVtZ2FYYco6Vv3XlLd2dZ-Xph6VXD04ch05hOwUL7vWm9pzf-Ob5d4JfSCDofUI7Xr2hsbt3AQ0dgaa57FlOmbXWcDvAhy5G5lOS3hCTV4I82AGgAooWuv0oXabueqeBKbuxGw6Jsh0fBzMgzJjNr2-SQbAB8PHsz3BbvRtF2gAacBn3nlzXb7VY1hKzhAjN0Vznridms96GD5XpEB27YorAo8b0cV1p4-UN_UFbuZNeri5p5mQX7eZ1o0Pl2EHZqyNRBnDdo_NpR_VhNfHo9VHXcBi2_tw_9ZRyNhRA7dfl240oe04V9SdNPQIAvqXPW-9AefusQoJILMhf8ygZdLOJoYP_eSE8Iwm90Fb0avGbdPjHB_DfKd20CrJQUWcIGWYiXmiVFWiSmlQ2PFdzT-NdM3LxSZDNzl8T4kBGAaCq_1scynr-JqN3DIuBZWmrZB1gYCuNtwOVZYqNUTSQNjQuJlFeqs_j7vW6FYSbRIwmjZ9wWHFSt_M9vheRTSa5nDqglmd8aNSSA03HU1UqHqjoV95DAyXezJmzqSw2VnncjMVRQubzzYz8RAResBZHqZN5xrlI9m715P0aHRFQY1rlc6XTGXOBr0Qhj5dHJByyP3n4TXB_V00KH27lK9zDAtX8jPmtAop1GGiqSxmv05Ctfe6THmcXQdMQIh0V8Xw0OngFtVLE7DFf2HnmQQ7weJ2_X2SaWVLW7rJ2M0zOGtwzVlQLtESWvqj79aG1VvgD-dn4lvnFNcIetIfwgJQFF94DDwVjBIu-CX5Y_NMRDxQghn2nsiHftIzJIc-H-bB29_kHnSsQc1pwrM3fG2JBV3_q54rLUChmWtLwfz_oktLJ8cskT5M78FjVcST1hnnoWacNcyOXKgFxu4DaIjpBCASL5J2lQ-6gIDKKlJzhE58ptm1O0eXrMt0CwnLQZXhLUZ3iOKuPqKmQdJsOl1lF_V-ZRvxco64tGDIMw523jszwBvPomQao32uu8DRqirkbMMZxOakt7WDtJKx1GQEQ8n8vYlxrdXiv0ycQeHrOpk6KblB3JcQ3hcBReEA3QwTJV-b5Vl3C3MBf5J3pFXP5AzOXgKpyX-ySQWbRTDhJL9nsd-mrxFsWmp3-GxsvQw9VPlkW76SmlCBRxIsTyk57F2Wi9U_r2brKtsGzEqykafimlQJmJAm7dHz25FYaUAkxk1TXHccPyVYV6jUN6rOk9-bxwznhE7oMThTxOjyRqDkSecv-EuUREm96XPn9QFJCIJKEuLOqx3MBGxILQHQkuIUEWjEhMdSbR8nQ2WT8mAI_S8iwcnymCf8lWdBb2z_edwUoWWRAvxkaq7aw3_53f3exvZKPVXsM3BhSYO9tId_lvzMVqnseU3HLZ7fFFWFuwBiZnD2BL-Gkt-_9O8Glfr1CF0ZrQDkXodrNkRWyfvSvrqYpclD-pWibj9_j-Y8JYxUvnWgbzfPDZ0xqQ6LQSzLYEB56R9EdgOjeWatNwYk_PMalxBNOl_pW_9fHGaNGUOteolOlFv90NtoLnfua1zKbm4IhuXMPXa4ARDm-z7uVgRuJ5umkpPey9DtOdMXank48OgZqrA-Yu1hBtEenY09wgFzkbDv9nIuH10tBdOSoyQOEAtXHEsVO0mTFd-QzUNLLuTZq9GtAgjq7ZtVe4rjo8pXISatB2fTTBMPpotphIKGFm1bFm17Bb3n1SH0cwksce6E7s_qDbT53jZ9inYkoMCCF9IxGLbw.FSUS53AXzGGGS0xQl-feJA"
}, conversation_id=None, parent_id=None)

print("Loading whisper engine...")
audio_model = whisper.load_model("base")

print("Starting speech recognition engine...")
r = sr.Recognizer()
r.energy_threshold = 300
r.pause_threshold = 0.8
r.dynamic_energy_threshold = False

# startMsg = "For the purpose of helping someone learn a new language, give human-like responses to all questions and ask questions in response. Act with a backstory, such as favorite foods, location of living, and more:\n"
startMsg = "Marv is a person that answers any questions and asks questions in the language of the user:\n"

print("Done! Ready to go!\n\n\n")

with sr.Microphone(sample_rate=16000) as source:
  while True:
    while True:
      print("\nSay something!..")
      audio = r.listen(source)
      torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
      output_text = audio_model.transcribe(torch_audio, language='spanish')["text"]
      print(output_text[1:])

      confirmation = str(input('\nConfirm text? (y/n): '))
      if (confirmation == 'Y' or confirmation == 'y'):
        break

    print("\nAsking the bot...")
    output_text = "User: "+output_text+"\n Marv:"
    response = chatbot.ask(startMsg+output_text, conversation_id=None, parent_id=None)['message']
    print(response)

    tts = gTTS(response, lang='es')
    mp3_file = BytesIO()
    tts.write_to_fp(mp3_file)
    mp3_file.seek(0)
    audio_segment = AudioSegment.from_file(mp3_file, format="mp3")
    play(audio_segment)