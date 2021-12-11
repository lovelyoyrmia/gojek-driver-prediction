import os
# TODO 1: Import flask libraries/dependencies
from flask import Flask, flash, redirect, render_template, url_for, request
from werkzeug.utils import secure_filename
import numpy as np
import tflite
from PIL import Image, ImageOps

# TODO 10: CONSTANTS
UPLOAD_FOLDER = 'static/uploads/'
# Allowed image extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
# Prediction threshold
PREDICTION_TRESHOLD = .4
# Comparison item
COMPARISON_ITEM = 'Gojek Driver'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# A function to check for allowed files
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Flask routing HTML pages 
@app.route('/')
def upload_form():
    return render_template('index.html')

# TODO 2: Flask routing HTML pages from a POST call
@app.route('/', methods=['POST'])
def upload_file():
    #check if the post request has the file part
    if 'file' not in request.files:
      flash('No file part')
      return redirect(request.url)
    file = request.files['file']
    
    # TODO 3 & 4: check for file and allowed file type
    if file and allowed_file(file.filename): 
      filename = secure_filename(file.filename)
      filepath = os.path.join(UPLOAD_FOLDER, filename)
      file.save(filepath)
      prediction = process_file(filepath)
      return render_template('index.html', filename=filename, prediction=prediction)

    else:
      return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

def process_file(filepath):
    # TODO 5: Load model, initialize tensor, get input and output tensors
    interpreter = tflite.Interpreter(model_path='model_unquant.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # TODO 6: Create the array of the right shape to feed into the keras model
    input_shape = input_details[0]['shape']
    input_data = np.ndarray(shape=input_shape, dtype=np.float32)
    image = Image.open(filepath)
    
    # TODO 7: Resize image to be at least 224x224 and then cropping from center, turn image to numpy array, normalize the image
    size = input_shape[1:3]
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32)/127.0 -1)
    
    # TODO 8: Load the image into the array, run the inference
    input_data[0] = normalized_image_array
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # TODO 9: Prediction text
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data

    prediction_text = truncate(prediction.item(0)*100, 2)
    if prediction.item(0) > PREDICTION_TRESHOLD:
      return f'Yay! {prediction_text}% a {COMPARISON_ITEM}'
    else:
      return f'{prediction_text}% NOT a {COMPARISON_ITEM}'


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


app.run(host='0.0.0.0', port=8080)