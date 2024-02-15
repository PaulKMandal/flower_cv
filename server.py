import flwr as fl

# Start Flower server
if __name__ == "__main__":
    # Define strategy (optional)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  # Sample 10% of available clients for training
        fraction_eval=0.1,  # Sample 10% of available clients for evaluation
        min_fit_clients=1,  # Minimum number of clients to be sampled for training
        min_eval_clients=1,  # Minimum number of clients to be sampled for evaluation
        min_available_clients=2,  # Minimum number of clients to be connected
    )

    # Start server with strategy
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 3},
        strategy=strategy,
    )