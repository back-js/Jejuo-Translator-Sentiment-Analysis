from flask import Flask, render_template, request, url_for, redirect
import os

from modules import interactive
from modules import translate
from modules import bert_sentiment
from modules.translate import bpencode, detok, tok
from flask_ngrok import run_with_ngrok
from datetime import datetime

gen = None

app = Flask(__name__)
run_with_ngrok(app)

port = int(os.environ.get("PORT", 5000))

@app.route('/')
def translate():
	return redirect("/translate/ne-en")

@app.route('/translate/ne-en')
def ne_en():
	global gen
	gen = interactive.Generator("/content/drive/My Drive/jejuo-translator/translator/app/model/je-ko", "/content/drive/My Drive/jejuo-translator/translator/app/model/je-ko.pt")
	return render_template("translate.html", title="제주어 감정분석기", active="translate", type="ne_en")

@app.route('/translate/ne-en/<string:sent>')
def ne_en_translate(sent):
    translated = detok(gen.generate(bpencode(tok(sent, lang="ne"), "ne_en")), lang="en")
    return render_template("transtext.html", data=translated)
    
@app.route('/translate/ne-en/sentiment/<string:sentence>')
def ne_en_sentiment(sentence):
    sentimented = bert_sentiment.predict_sentiment(sentence)
    return render_template("sentiment.html", data=sentimented) 

@app.route('/translate/en-ne')
def en_ne():
	global gen
	gen = interactive.Generator("/content/drive/My Drive/jejuo-translator/translator/app/model/ko-je", "/content/drive/My Drive/jejuo-translator/translator/app/model/ko-je.pt")
	return render_template("translate.html", title="제주어 기계번역기", active="translate", type="en_ne")

@app.route('/translate/en-ne/<string:sent>')
def en_ne_translate(sent):
	translated = detok(gen.generate(bpencode(tok(sent, lang="en"), "en_ne")), lang="ne")	
	return render_template("transtext.html", data=translated)


if __name__ == '__main__':
    app.run()