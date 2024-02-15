import flwr as fl
from model import get_model
from dataset import get_dataloader
from utils import train, test

class ObjectDetectionClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = get_model()
        self.train_loader, self.test_loader = get_dataloader()

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.train_loader)
        return self.get_parameters(), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_loader)
        return float(loss), len(self.test_loader), {"accuracy": float(accuracy)}

# Start Flower client
if __name__ == "__main__":
    fl.client.start_client(server_address="[::]:8080", client=ObjectDetectionClient())
