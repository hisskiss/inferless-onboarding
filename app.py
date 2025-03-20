import torch
from transformers import pipeline
from pydantic import BaseModel, Field
import inferless

@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="a horse near a beach")

@inferless.response
class ResponseObjects(BaseModel):
    generated_txt: str = Field(default='Test output')

app = inferless.Cls(gpu="T4")
class InferlessPythonModel:
    @app.load
    def initialize(self):
        self.generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M",device=0)
    @app.infer
    def infer(self, inputs):
        pipeline_output = self.generator(inputs.prompt, do_sample=True, min_length=128)
        generateObject = ResponseObjects(generated_txt = pipeline_output[0]["generated_text"])
        return generateObject


@inferless.local_entry_point
def my_local_entry(dynamic_params):
    from app import InferlessPythonModel 
    model_instance = InferlessPythonModel()
    return model_instance.infer(RequestObjects(dynamic_params))
