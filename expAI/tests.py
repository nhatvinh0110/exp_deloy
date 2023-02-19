# import subprocess

# print(subprocess.run(["python3", "./AI_models/Pytorch_Retinaface/train.py"],
#                      capture_output=True))


from AI_models.Pytorch_Retinaface.train2 import train_mnet

json_config = {
    'name': 'mobilenet0.25',
    'save_folder': './AI_models/Pytorch_Retinaface/weights/',
    'gpu_train': True,
    'batch_size': 16,
    'ngpu': 1,
    'epoch': 250,
    'image_size': 640
}

# {
#     "name": "mobilenet0.25",
#     "save_folder": "./expAI/AI_models/Pytorch_Retinaface/weights/",
#     "gpu_train": "True",
#     "batch_size": 16,
#     "ngpu": 1,
#     "epoch": 250,
#     "image_size":640
# }

train_mnet(1,'./AI_models/Pytorch_Retinaface/data/widerface/train/label.txt',json_config)



# import threading
# t = threading.Thread(target=train,
#                         args=('/',"{}"), kwargs={})
# t.setDaemon(True)
# t.start()