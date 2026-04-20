from flask import Flask, render_template, Response, request
import web_simulator

app = Flask(__name__)

import os
from flask import send_from_directory

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reports/<patient_id>')
def view_report(patient_id):
    # reports are saved in the 'reports' folder by run2.py
    report_dir = os.path.join(os.path.dirname(__file__), 'reports')
    filename = f"report_{patient_id}.html"
    return send_from_directory(report_dir, filename)

@app.route('/stream')
def stream():
    scenario = request.args.get('scenario', default=1, type=int)
    fast = request.args.get('fast', default='true').lower() == 'true'
    
    # SSE Stream
    return Response(
        web_simulator.run_simulation_stream(scenario_id=scenario, fast=fast), 
        mimetype='text/event-stream'
    )

if __name__ == '__main__':
    # Use threaded=True handles SSE concurrent connections properly
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
