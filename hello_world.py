import runpod

def handler(job):
    """
    This is a simple handler that takes a name as input and returns a greeting.
    The job parameter contains the input data in job["input"]
    """
    job_input = job["input"]
    print(job_input)
    name = job_input.get("name", "World")
    return f"Hello, {name}! Welcome to RunPod Serverless!"

runpod.serverless.start({"handler": handler})