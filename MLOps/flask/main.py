from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def base_route():
    return "<h1>Home Page</h1>"

@app.route("/about")
def about_page():
    return "<h2>About Page</h2>"

@app.route("/login")
def login_page():
    return "<h3>Logging In</h3>"

@app.route("/user/<user_id>")
def user_info(user_id):
    return f"<h3>This user:</h3><h2>{user_id}</h2>"

def square(num):
    return int(num)*int(num)

@app.route("/square", methods=['GET'])
def run_templates():
    if request.method == 'GET':
        if request.args.get('num') == '':
            return f"<h2>INVALID INPUT</h2>"
        elif request.args.get("num") == None:
            return render_template("square.html")
        else:
            num = request.args.get("num")
            return render_template("solution.html", num=num, squareofnum=square(num))
