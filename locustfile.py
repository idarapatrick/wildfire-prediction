import random
from locust import HttpUser, task, between

# Railway URLS, serving as load-balanced endpoints(Imitating Docker containers)
SERVERS = [
    "https://wildfire-prediction.up.railway.app",
    "https://wildfire-prediction-2.up.railway.app",
    "https://wildfire-prediction-3.up.railway.app"
]

class WildfireUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_image(self):
        # Randomly pick one of the 3 containers
        target_server = random.choice(SERVERS)
        
        # Construct the full URL
        endpoint = f"{target_server}/predict"
        
        # Send the image
        try:
            with open("data/test/wildfire/-78.2725,49.4581.jpg", "rb") as image:
                # use the full URL to override the host
                self.client.post(endpoint, files={"file": image})
        except FileNotFoundError:
            print("Error: Image file not found. Check path.")

    @task(3)
    def check_status(self):
        target_server = random.choice(SERVERS)
        self.client.get(f"{target_server}/status")