from flask import Flask
import hello

app = Flask(__name__)

@app.route('/')
def home():
    return hello.main()

if __name__ == "__main__":
    app.run(host='0.0.0.0')