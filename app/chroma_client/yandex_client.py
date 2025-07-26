from unittest import result
from torch.cuda import temperature
from yandex_cloud_ml_sdk import YCloudML


class YandexGPTWrapper:
    def __init__(self, folder_id, api_key, model_name="yandexgpt"):
        self.sdk = YCloudML(folder_id=folder_id, auth=api_key)
        self.model = self.sdk.models.completions(model_name)

    def invoke(self, messages):
        operation = self.model.configure(temperature=0.7).run_deferred(messages)
        result = operation.wait()
        return result
