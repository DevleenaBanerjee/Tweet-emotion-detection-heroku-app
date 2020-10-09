from predict.predict_emotion import result

import os
from flask import render_template
from flask import Flask,request
app = Flask(__name__)

@app.route('/')
def load_root():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def get_prediction():
        if request.method == 'POST':
            tweet = request.form.get('tweet')
            tod = request.form.get('tod')
        json=result(tweet,tod)
        return json
        
if __name__ == "__main__":
    app.run(int(os.environ.get('PORT', 5000)),debug=True)

