from roboflow import Roboflow
rf = Roboflow(api_key="Yn0X3uWJ1PT4KqePZjkT")
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(10)
dataset = version.download("yolov8")
                