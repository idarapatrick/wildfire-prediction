from locust import HttpUser, task, between

class WildfireUser(HttpUser):
    wait_time = between(1, 3) # Users wait 1-3 seconds between actions

    @task
    def predict_image(self):
        with open("data/test/wildfire/-78.375,51.2861.jpg", "rb") as image:
            self.client.post("/predict", files={"file": image})

    @task(3) 
    def check_status(self):
        self.client.get("/status")