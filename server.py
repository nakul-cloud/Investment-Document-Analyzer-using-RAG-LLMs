# server.py
from flask import Flask, render_template, request, jsonify
from app import InvestmentAnalyzer
import os

app = Flask(__name__)

# Global analyzer instance
analyzer = InvestmentAnalyzer()
analyzer.setup_llm()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "pdf" not in request.files:
            return jsonify({"status": "error", "msg": "No file field 'pdf' in request"})

        file = request.files["pdf"]
        if file.filename == "":
            return jsonify({"status": "error", "msg": "No selected file"})

        filepath = os.path.join(UPLOAD_DIR, file.filename)
        file.save(filepath)

        ok = analyzer.process_pdf(filepath)
        if not ok:
            return jsonify({"status": "error", "msg": "Failed to extract text from PDF"})

        return jsonify({"status": "ok"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "msg": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        question = (data or {}).get("question", "").strip()
        if not question:
            return jsonify({"status": "error", "msg": "Question is empty"}), 400

        result = analyzer.retrieve_answer(question)
        return jsonify({
            "status": "ok",
            "answer": result["answer"],
            "pages": result["pages"],
            # optional: send context if you want to show it
            # "context": result["context_chunks"],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "msg": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
