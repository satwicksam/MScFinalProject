from flask import *
import whisper
import librosa
import moviepy.editor as mp
from textblob import TextBlob as tb
import pycountry
import zipfile36
import os

app=Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("SpeechRecognition.html")

@app.route("/speech-recognition", methods=["POST", "GET"])
def speechRecognition():
    if request.method == "POST":
        if request.form["run"] == "Recognize":
            if request.form["choice"] == "ar":
                f = request.files["upfile"]
           
                f.save("static/uploads_audio/"+f.filename)
                audio_src = ("static/uploads_audio/"+f.filename)
                
                # f.save(os.path.join(uploads_dir, secure_filename(f.filename)))
                # audio_src = os.path.join(uploads_dir, secure_filename(f.filename))

                model = whisper.load_model("base")
                audio, sr = librosa.load(audio_src)
                
                audio_la = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio_la).to(model.device)
                _, probs = model.detect_language(mel)
                audio_language = display_language_name(max(probs, key=probs.get))
                lang = f"Detected Audio Language : {audio_language}"
                
                result=model.transcribe(audio)
                text = result["text"]
                
                ############# SENTIMENT #############
                
                stext=tb(text)
                sentiment_score = stext.sentiment.polarity
                
                if sentiment_score > 0:
                    sentiment = "Positive"
                elif sentiment_score < 0:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                    
                sentiment = f"Detected Sentiment : {sentiment}"
                
                ############# SENTIMENT #############
                
                return render_template("SpeechRecognition.html", language=lang, text=text, sentiment=sentiment, media=url_for("static", filename="uploads_audio/"+f.filename))
            
            ####################################################################################################
            
            elif request.form["choice"] == "vr":
                f=request.files["upfile"]
                f.save("static/uploads_video/"+f.filename)
                
                clip = mp.VideoFileClip(r"static/uploads_video/"+f.filename)
                clip.audio.write_audiofile(r"static/uploads_video/audio.mp3")
                
                audio_src = ("static/uploads_video/audio.mp3")
                
                model = whisper.load_model("base")
                audio, sr = librosa.load(audio_src)
                
                audio_la = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio_la).to(model.device) 
                _, probs = model.detect_language(mel)
                audio_language = display_language_name(max(probs, key=probs.get))
                lang=f"Detected Video Language : {audio_language}"
                
                result=model.transcribe(audio)
                text = result["text"]
                
                ############# SENTIMENT #############
                
                stext=tb(text)
                sentiment_score = stext.sentiment.polarity
                
                if sentiment_score > 0:
                    sentiment = "Positive"
                elif sentiment_score < 0:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                    
                sentiment = f"Detected Sentiment : {sentiment}"
                
                ############# SENTIMENT #############
                
                return render_template("SpeechRecognition.html", language=lang, text=text, sentiment=sentiment, media=url_for("static", filename="uploads_video/"+f.filename))
            
        # if request.form["bmp"] == "Back To SPEECH RECOGNITION":
        #     return render_template("SpeechRecognition.html")


#################### RECORD VOICE ####################
@app.route("/record-voice", methods=["POST", "GET"])
def rv():
    if request.method == "POST":
        if request.form["rv"] == "Click Here":
            return render_template("record.html")
        
#################### MULTIPLE FILE USER INTERFACE ####################
@app.route("/srmf", methods=["POST", "GET"])
def SRMF():
    if request.method == "POST":
        if request.form["srmf"] == "Click Here":
            return render_template("multipleSR.html")
        
#################### MULTIPLE FILE ####################
@app.route("/speech-recognition-multiple", methods=["POST", "GET"])
def speechRecognitionMF():
    text1 = ""
    if request.method == "POST":
        if request.form["run"] == "Recognize":
            f = request.files["upfile"]
           
            f.save("static/uploads_audio_zip/"+f.filename)
            audio_src = ("static/uploads_audio_zip/"+f.filename)

            model = whisper.load_model("base")

            for fn in os.listdir("static/extract_audio/"):
                os.remove("static/extract_audio/"+fn)

            with zipfile36.ZipFile(audio_src, "r") as zf:
                zf.extractall("static/extract_audio/")

            for fn in os.listdir("static/extract_audio/"):
                ars = ("static/extract_audio/"+fn)
                audio, sr = librosa.load(ars)
                audio_la = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio_la).to(model.device)
                _, probs = model.detect_language(mel)
                audio_language = display_language_name(max(probs, key=probs.get))
                lang = f"Detected Audio Language : {audio_language}"
                    
                result=model.transcribe(audio)
                text = result["text"]
                
            ############# SENTIMENT #############
                
                stext=tb(text)
                sentiment_score = stext.sentiment.polarity
                    
                if sentiment_score > 0:
                    sentiment = "Positive"
                elif sentiment_score < 0:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                        
                sentiment = f"Detected Sentiment : {sentiment}"
                
            ############# SENTIMENT #############
                text1 += fn + ":" + result["text"] + "--" + sentiment + "\n\n"

            return render_template("multipleSR.html", text=text1)
            
#################### MAIN ####################
if __name__=="__main__":
    app.run()

####### FUNCTIONS #######

def display_language_name(language_code):
    try:
        language = pycountry.languages.get(alpha_2=language_code)
        return language.name
    except:
        return "Unknown"