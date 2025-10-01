# app.py
from flask import Flask, jsonify, send_from_directory, request
from query import search  # reuse search() from query.py
import os

app = Flask(__name__)
IMAGES_DIR = os.path.join("output","images")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    q = data.get("question", "")
    results = search(q, k=4)
    # convert image paths to URLs
    for r in results:
        if r.get("image"):
            fname = os.path.basename(r["image"])
            r["image_url"] = request.host_url.rstrip("/") + "/images/" + fname
        else:
            r["image_url"] = None
    return jsonify({"question": q, "results": results})

@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(IMAGES_DIR, filename)

if __name__=="__main__":
    app.run(debug=True, port=5000)
