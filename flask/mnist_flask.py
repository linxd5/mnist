#!/usr/bin/env python
# coding=utf-8

from flask import Flask, render_template
import os

wrong_images_num = 0

wrong_images_info = []
for wrong_image in os.listdir('./static/wrong_images/'):
    wrong_image_info = {
        'name': 'wrong_images/' + wrong_image,
        'predict_value': wrong_image.split('_')[0],
        'actual_value': wrong_image.split('_')[1]
    } 
    wrong_images_info.append(wrong_image_info)
    wrong_images_num += 1

# Initialize the Flask application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', 
                            wrong_images_num = wrong_images_num,
                            wrong_images_info = wrong_images_info)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
