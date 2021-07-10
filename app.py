from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask import jsonify
import os

import config
import predict_2, predict_3

app = Flask(__name__)
app.config.from_object(config)
db = SQLAlchemy(app)
# db = SQLAlchemy(app)
Upload_Folder = "./static/predict_image/unpredict"


# class Item(db.Model):
#     id = db.Column(db.Integer(), primary_key=True)
#     name = db.Column(db.String(length=30), nullable=False, unique=True)
#     price = db.Column(db.Integer(), nullable=False, unique=True)
#     barcode = db.Column(db.String(length=12), nullable=False, unique=True)
#     description = db.Column(db.String(length=1024), nullable=False, unique=True)
class doctor(db.Model):
    __tablename__ = 'doctor'
    id = db.Column(db.INTEGER(), primary_key=True)
    name = db.Column(db.String(length=255), nullable=False, unique=True)
    hospital = db.Column(db.String(length=255), nullable=False, unique=True)
    link = db.Column(db.String(length=255), nullable=False, unique=True)
    title = db.Column(db.String(length=255), nullable=False, unique=True)
    skills = db.Column(db.String(length=255), nullable=False, unique=True)
    price = db.Column(db.String(length=255), nullable=False, unique=True)


def __init__(self, name, hospital, link, title, skills, price):
    self.name = name
    self.hospital = hospital
    self.link = link
    self.title = title
    self.skills = skills
    self.price = price


@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')


@app.route('/cancer', methods=["GET", "POST"])
def cancer_page():
    return render_template('test.html', prediction=0)


@app.route('/upload', methods=["GET", "POST"])
def upload_pic():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            for f in os.listdir(Upload_Folder):
                path_file2 = os.path.join(Upload_Folder, f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)
            image_location = os.path.join(
                Upload_Folder,
                image_file.filename
            )
            image_file.save(image_location)

            return render_template('home.html', upload=1, imagename=image_file.filename)
    return render_template('home.html', upload=0)

@app.route('/predict', methods=["GET", "POST"])
def pre():
    label_list = {}
    predict_batch = predict_2.get_inputs("./static/predict_image")
    label_of_y = predict_2.predict(predict_batch)
    label_list['predict_2'] = label_of_y

    test_batch = predict_3.pre_imput("./static/predict_image")

    label_of_y = predict_3.predict_image(test_batch)
    label_list['predict_3'] = label_of_y
    return jsonify(label_list)

@app.route('/market')
def market_page():
    # items = [
    #     {'id': 1, 'name': 'Phone', 'barcode': '893212299897', 'price': 500},
    #     {'id': 2, 'name': 'Laptop', 'barcode': '123985473165', 'price': 900},
    #     {'id': 3, 'name': 'Keyboard', 'barcode': '231985128446', 'price': 150}
    # ]
    doctors = doctor.query.all()
    doctorlist = []
    for d in doctors:
        dict = {}
        dict['id'] = d.id
        dict['name'] = d.name
        dict['hospital'] = d.hospital
        dict['skills'] = d.skills
        dict['title'] = d.title
        dict['price'] = d.price
        doctorlist.append(dict)

    return render_template('market.html', items=doctorlist)

@app.route('/damn')
def damn():
    # items = [
    #     {'id': 1, 'name': 'Phone', 'barcode': '893212299897', 'price': 500},
    #     {'id': 2, 'name': 'Laptop', 'barcode': '123985473165', 'price': 900},
    #     {'id': 3, 'name': 'Keyboard', 'barcode': '231985128446', 'price': 150}
    # ]
    doctor_list = doctor.query.all()
    for d in doctor_list:
        dict = {}

        print(type(d.id))
    print(type(doctor_list[0]))
    return doctor_list[0]


if __name__ == '__main__':
    app.run()
