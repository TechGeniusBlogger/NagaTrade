# Kronos Forecast - desktop app. Opens in your browser at http://localhost:8767.
# First run: accept the agreement, enter your own Dhan client id + access token (stored only on this
# computer, sent only to api.dhan.co), download the Kronos model once. This software never places an
# order. Educational tool, not investment advice.
import json
import os
import sys
import threading
import time
import webbrowser

from flask import Flask, Response, jsonify, redirect, request, send_from_directory

import service as S

VERSION = "1.0.0"
PORT = 8767
ASSETS = os.path.join(S.BASE, "assets")
CFG = os.path.join(S.HOME, "config.json")

cfg = {"agreed_at": "", "agreed_version": "", "client_id": "", "access_token": ""}
if os.path.exists(CFG):
    try:
        cfg.update(json.load(open(CFG)))
    except ValueError:
        pass
S.set_keys(cfg["client_id"], cfg["access_token"])


def save_cfg():
    with open(CFG, "w") as f:
        json.dump(cfg, f, indent=1)
    try:
        os.chmod(CFG, 0o600)
    except OSError:
        pass


def agreed():
    return bool(cfg["agreed_at"]) and cfg["agreed_version"] == VERSION


def ready():
    S.refresh_model_state()
    return agreed() and bool(cfg["client_id"] and cfg["access_token"]) and S.model_state["status"] == "ready"


app = Flask(__name__)


def err(msg, code):
    return Response(json.dumps({"error": msg}), status=code, mimetype="application/json")


def json_post():
    # State-changing calls must be JSON: a web page from another site can't send that to this
    # local server without a CORS pre-flight, which this server never approves.
    return request.method == "POST" and request.is_json


@app.route("/")
def index():
    if not ready():
        return redirect("/setup")
    return send_from_directory(ASSETS, "kronos_forecast.html")


@app.route("/setup")
def setup():
    return send_from_directory(ASSETS, "setup.html")


@app.route("/mentorship/d3.min.js")
def d3():
    return send_from_directory(ASSETS, "d3.min.js")


@app.route("/api/state")
def state():
    S.refresh_model_state()
    m = S.model_state
    return jsonify(version=VERSION, agreed=agreed(), agreement=open(os.path.join(ASSETS, "agreement.txt"), encoding="utf-8").read(),
                   has_keys=bool(cfg["client_id"] and cfg["access_token"]), client_id=cfg["client_id"],
                   model=dict(status=m["status"], done=m["done"], total=m["total"], error=m["error"]),
                   ready=ready(), home=S.HOME)


@app.route("/api/agree", methods=["POST"])
def agree():
    if not json_post():
        return err("bad request", 400)
    cfg["agreed_at"], cfg["agreed_version"] = time.strftime("%Y-%m-%dT%H:%M:%S"), VERSION
    save_cfg()
    return jsonify(ok=True)


@app.route("/api/keys", methods=["POST"])
def keys():
    if not json_post():
        return err("bad request", 400)
    body = request.get_json(silent=True) or {}
    cid, tok = str(body.get("client_id", "")).strip(), str(body.get("token", "")).strip()
    if not cid or not tok:
        return err("Enter both your Dhan client id and access token.", 400)
    try:
        S.check_keys(cid, tok)
    except S.DhanAuthError:
        return err("Dhan didn't accept these keys. Check the client id and paste a fresh access token.", 400)
    except Exception as e:
        return err(f"Couldn't reach Dhan to check the keys: {str(e)[:150]}", 502)
    cfg["client_id"], cfg["access_token"] = cid, tok
    save_cfg()
    S.set_keys(cid, tok)
    return jsonify(ok=True)


@app.route("/api/model/download", methods=["POST"])
def model_download():
    if not json_post():
        return err("bad request", 400)
    if not agreed():
        return err("Accept the agreement first.", 409)
    threading.Thread(target=S.download_model, daemon=True).start()
    return jsonify(ok=True)


def guarded(fn):
    if not ready():
        return err("Finish setup first.", 409)
    sym = request.args.get("symbol", "CIPLA").strip().upper()
    if sym not in S.universe():
        return err("not in the list: Nifty 50, Sensex and NSE F&O stocks only", 404)
    try:
        resp = jsonify(fn(sym))
    except S.DhanAuthError as e:
        return err(str(e), 401)
    except (ValueError, RuntimeError, OSError) as e:
        return err(str(e)[:300], 502)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/kronos/universe")
def universe():
    return jsonify([{"symbol": u["symbol"], "name": u["name"]} for u in S.universe().values()])


@app.route("/kronos/forecast/data")
def forecast_data():
    return guarded(S.payload)


@app.route("/kronos/forecast/one")
def forecast_one():
    return guarded(lambda sym: S.forecast_at(sym, request.args.get("k", "0")))


def main():
    url = f"http://localhost:{PORT}"
    print(f"Kronos Forecast {VERSION} -> {url}   (close this window to stop)")
    print(f"Your settings and data live in: {S.HOME}")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
