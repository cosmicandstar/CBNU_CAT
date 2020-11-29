from django.apps import AppConfig

model = None
isLaunched = False

def getModel():
    return model

class MainConfig(AppConfig):
    name = 'main'

    def ready(self):
        getModel()
        print("처음만 실행")
