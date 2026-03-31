from flask import Flask, request, render_template
import os
from utils import predict

# ✅ Create app FIRST
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Then define route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        preds = predict(filepath)

        result = []
        for label, desc, prob in preds:
            result.append(f"{desc} : {round(prob * 100, 2)}%")

        img_path = filepath

    return render_template("index.html", result=result, img_path=img_path)

# ✅ Run app LAST
if __name__ == "__main__":
    app.run(debug=True)