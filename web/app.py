import os
import sys
import csv
import cv2
import torch
import numpy as np
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    Response
)

# ==================================================
# FLASK CONFIG
# ==================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

RESULT_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "results"
)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "history.csv"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# ==================================================
# IP CAMERA
# ==================================================

IP_WEBCAM_URL = "http://192.168.137.21:4747/video"

# ==================================================
# LOAD YOLOV7
# ==================================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            BASE_DIR,
            ".."
        )
    )
)

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (
    non_max_suppression,
    scale_coords
)
from utils.torch_utils import select_device

device = select_device("0")

weights = os.path.join(
    "runs",
    "train",
    "exp4",
    "weights",
    "best.pt"
)

model = attempt_load(
    weights,
    map_location=device
)

model.eval()

stride = int(model.stride.max())

names = (
    model.module.names
    if hasattr(model, "module")
    else model.names
)

# ==================================================
# HISTORY CSV
# ==================================================

if not os.path.exists(HISTORY_FILE):

    with open(
        HISTORY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Tanggal",
            "Jenis",
            "Jumlah Objek",
            "Babi Hutan",
            "Tikus",
            "Confidence"
        ])
# ==================================================
# DETECT IMAGE
# ==================================================

def detect_image(img):

    img0 = img.copy()

    img = letterbox(
        img0,
        640,
        stride=stride,
        auto=True
    )[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    pred = non_max_suppression(
        pred,
        conf_thres=0.50,
        iou_thres=0.45
    )

    detections = []

    total_babi = 0
    total_tikus = 0

    highest_conf = 0

    for det in pred:

        if len(det):

            det[:, :4] = scale_coords(
                img.shape[2:],
                det[:, :4],
                img0.shape
            ).round()

            for *xyxy, conf, cls in reversed(det):

                cls = int(cls)

                label = names[cls]

                confidence = round(float(conf) * 100, 2)

                detections.append({

                    "label": label,
                    "confidence": confidence

                })

                if confidence > highest_conf:
                    highest_conf = confidence

                if "babi" in label.lower():
                    total_babi += 1

                elif "tikus" in label.lower():
                    total_tikus += 1

                x1 = int(xyxy[0])
                y1 = int(xyxy[1])
                x2 = int(xyxy[2])
                y2 = int(xyxy[3])

                cv2.rectangle(
                    img0,
                    (x1, y1),
                    (x2, y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    img0,
                    f"{label} {confidence:.1f}%",
                    (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0,255,0),
                    2
                )

    summary = {

        "total": total_babi + total_tikus,

        "babi": total_babi,

        "tikus": total_tikus,

        "highest_confidence": highest_conf

    }

    return img0, detections, summary


# ==================================================
# HISTORY
# ==================================================

def save_history(jenis, summary):

    tanggal = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    with open(
        HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            tanggal,

            jenis,

            summary["total"],

            summary["babi"],

            summary["tikus"],

            summary["highest_confidence"]

        ])


def load_history():

    histories = []

    if os.path.exists(HISTORY_FILE):

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                histories.append(row)

    histories.reverse()

    return histories

# ==================================================
# ROUTES
# ==================================================

@app.route("/")
def dashboard():

    return render_template(
        "dashboard.html"
    )


@app.route("/image")
def image():

    return render_template(
        "image.html",
        original_image=None,
        image_file=None,
        detections=[],
        summary=None
    )


@app.route("/video")
def video():

    return render_template(
        "video.html",
        video_filename=None,
        summary=None
    )


@app.route("/realtime")
def realtime_page():

    return render_template(
        "realtime.html"
    )


@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


@app.route("/history")
def history():

    return render_template(
        "history.html",
        histories=load_history()
    )


# ==================================================
# IMAGE UPLOAD
# ==================================================

@app.route("/upload_image", methods=["POST"])
def upload_image():

    file = request.files.get("file")

    if file is None or file.filename == "":

        return redirect(url_for("image"))

    filename = file.filename

    upload_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(upload_path)

    img = cv2.imread(upload_path)

    result_img, detections, summary = detect_image(img)

    result_name = "result_" + filename

    result_path = os.path.join(
        app.config["RESULT_FOLDER"],
        result_name
    )

    cv2.imwrite(
        result_path,
        result_img
    )

    save_history(
        "Gambar",
        summary
    )

    return render_template(

        "image.html",

        original_image="uploads/" + filename,

        image_file="results/" + result_name,

        detections=detections,

        summary=summary

    )


# ==================================================
# VIDEO UPLOAD
# ==================================================

@app.route("/upload_video", methods=["POST"])
def upload_video():

    file = request.files.get("file")

    if file is None or file.filename == "":

        return redirect(url_for("video"))

    filename = file.filename

    video_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(video_path)

    return render_template(

        "video.html",

        video_filename=filename,

        summary=None

    )


# ==================================================
# VIDEO STREAM
# ==================================================

def gen_video(video_path):

    cap = cv2.VideoCapture(video_path)

    total_babi = 0

    total_tikus = 0

    highest_conf = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:

            break

        result_img, detections, summary = detect_image(frame)

        total_babi += summary["babi"]

        total_tikus += summary["tikus"]

        if summary["highest_confidence"] > highest_conf:

            highest_conf = summary["highest_confidence"]

        ret, buffer = cv2.imencode(
            ".jpg",
            result_img
        )

        if not ret:

            continue

        frame = buffer.tobytes()

        yield (

            b'--frame\r\n'

            b'Content-Type: image/jpeg\r\n\r\n'

            + frame +

            b'\r\n'

        )
    summary = {

    "total": total_babi + total_tikus,

    "babi": total_babi,

    "tikus": total_tikus,

    "highest_confidence": highest_conf

    }

    save_history(

        "Video",

        summary

    )

    cap.release()

# ==================================================
# STREAM VIDEO ROUTE
# ==================================================

@app.route("/stream_video/<filename>")
def stream_video(filename):

    video_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    return Response(
        gen_video(video_path),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ==================================================
# REALTIME
# ==================================================

def gen_frames(source):

    cap = cv2.VideoCapture(source)
    total_babi = 0

    total_tikus = 0

    highest_conf = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        result_img, detections, summary = detect_image(frame)

        total_babi += summary["babi"]
        total_tikus += summary["tikus"]

        if summary["highest_confidence"] > highest_conf:
            highest_conf = summary["highest_confidence"]

        ret, buffer = cv2.imencode(
            ".jpg",
            result_img
        )

        if not ret:
            continue

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
         + frame +
         b'\r\n'
        )


    summary = {
        "total": total_babi + total_tikus,
        "babi": total_babi,
        "tikus": total_tikus,
        "highest_confidence": highest_conf
    }

    save_history(
        "Realtime",
        summary
    )

    cap.release()


# ==================================================
# REALTIME PC
# ==================================================

@app.route("/realtime_pc")
def realtime_pc():

    return Response(

        gen_frames(0),

        mimetype="multipart/x-mixed-replace; boundary=frame"

    )


# ==================================================
# REALTIME HP
# ==================================================

@app.route("/realtime_hp")
def realtime_hp():

    return Response(

        gen_frames(IP_WEBCAM_URL),

        mimetype="multipart/x-mixed-replace; boundary=frame"

    )


# ==================================================
# UPDATE IP CAMERA
# ==================================================

@app.route("/set_camera", methods=["POST"])
def set_camera():

    global IP_WEBCAM_URL

    ip = request.form.get("ip")

    if ip is not None and ip.strip() != "":

        IP_WEBCAM_URL = ip.strip()

    return redirect(
        url_for("realtime_page")
    )


# ==================================================
# DOWNLOAD HISTORY (CSV)
# ==================================================

@app.route("/download_history")
def download_history():

    return Response(

        open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ),

        mimetype="text/csv",

        headers={

            "Content-Disposition":
            "attachment; filename=history.csv"

        }

    )


# ==================================================
# CLEAR HISTORY
# ==================================================

@app.route("/clear_history")
def clear_history():

    with open(
        HISTORY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            "Tanggal",

            "Jenis",

            "Jumlah Objek",

            "Babi Hutan",

            "Tikus",

            "Confidence"

        ])

    return redirect(
        url_for("history")
    )


# ==================================================
# ERROR PAGE
# ==================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "404.html"
    ), 404


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    os.makedirs(
        app.config["UPLOAD_FOLDER"],
        exist_ok=True
    )

    os.makedirs(
        app.config["RESULT_FOLDER"],
        exist_ok=True
    )

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )