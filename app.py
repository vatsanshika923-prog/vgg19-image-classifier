from flask import Flask, request, render_template
import os
from utils import predict

# ✅ Create app
app = Flask(__name__)

# ✅ Configurations
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# ✅ Route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_path = None

    if request.method == "POST":
        # ✅ Check if file exists
        if "file" not in request.files:
            return render_template("index.html", result=None, img_path=None)

        file = request.files["file"]

        # ✅ Check if filename is empty
        if file.filename == "":
            return render_template("index.html", result=None, img_path=None)

        # ✅ Ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # ✅ Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # ✅ Predict
        preds = predict(filepath)

        # ✅ Format output
        result = []
        for label, desc, prob in preds:
            result.append(f"{desc} : {round(prob * 100, 2)}%")

        img_path = filepath

    return render_template("index.html", result=result, img_path=img_path)


# ✅ Run app
if __name__ == "__main__":
    app.run(debug=True)