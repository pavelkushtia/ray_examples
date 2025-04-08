import ray
from ray import serve
import requests

# Start Ray
ray.init()

# Start Serve
serve.start()

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class Converter:
    async def __call__(self, request):
        if request.query_params["type"] == 'CF' :
            return {"Fahrenheit temperature":
            9.0/5.0 * float(request.query_params["temp"]) + 32.0}
        elif request.query_params["type"] == 'FC' :
            return {"Celsius temperature":
            (float(request.query_params["temp"]) - 32.0) * 5.0/9.0 }
        else:
            return {"Unknown conversion code" : request.query_params["type"]}

translator_app = Converter.bind()

# ray.shutdown()