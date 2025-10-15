import numpy as np
import pandas as pd

def generate_poisson_timestamps(rate: float, num_events: int) -> np.ndarray:
    """
    Generate a set of timestamps following a Poisson distribution.

    :param rate: Event rate (lambda) per unit time.
    :param num_events: The number of events to generate.
    :return: A numpy array of timestamps.
    """
    # Generate inter-arrival times following an exponential distribution
    inter_arrival_times = np.random.exponential(scale=1/rate, size=num_events)
    # Compute the timestamps
    timestamps = np.cumsum(inter_arrival_times)
    return timestamps

def generate_gamma_timestamps(shape: float, scale: float, num_events: int) -> np.ndarray:
    """
    Generate a set of timestamps following a Gamma distribution.

    :param shape: Shape parameter (k) of the Gamma distribution.
    :param scale: Scale parameter (theta) of the Gamma distribution.
    :param num_events: The number of events to generate.
    :return: A numpy array of timestamps.
    """
    # Generate inter-arrival times following a Gamma distribution
    inter_arrival_times = np.random.gamma(shape, scale, num_events)
    # Compute the timestamps
    timestamps = np.cumsum(inter_arrival_times)
    return timestamps

def analysis_request_rate(timestamps: np.ndarray):
    # Convert timestamps to pandas datetime type
    timestamps_pd = pd.to_datetime(timestamps, unit='s')  # Assuming timestamps are in nanoseconds

    # Create a DataFrame
    df = pd.DataFrame(timestamps_pd, columns=["Timestamp"])

    # Sort timestamps
    df = df.sort_values('Timestamp')

    # Convert timestamps to Unix time in nanoseconds
    df['Unix_Nanoseconds'] = df['Timestamp'].astype(np.int64) // 1e9

    # Group timestamps into 1-second intervals
    df['1s_bin'] = df['Unix_Nanoseconds'] // 1

    # Count the number of requests in each interval
    request_counts = df.groupby('1s_bin').size().reset_index(name='Request_Count')

    # Calculate the request rate (requests per second)
    request_counts['Request_Rate'] = request_counts['Request_Count']

    # Print the mean and variance of the request rate
    print(np.mean(request_counts['Request_Rate']), np.var(request_counts['Request_Rate']))

def save_timestamps_to_csv(timestamps: np.ndarray, filename: str):
    """
    Save timestamps to a CSV file.

    :param timestamps: A numpy array of timestamps.
    :param filename: The name of the CSV file.
    """
    df = pd.DataFrame(timestamps, columns=["Timestamp"])
    df.to_csv(filename, index=False)

# Parameter setup
request_rate = 5
coefficient_of_variation = 2  # Coefficient of variation, 1 for Poisson distribution, higher values mean higher variance

poisson_para = request_rate  # Poisson distribution's average event rate per unit time (request rate)
gamma_mean = 1 / request_rate  # Mean of the Gamma distribution, the reciprocal of the request rate
gamma_var = (gamma_mean * coefficient_of_variation) ** 2  # Variance of the Gamma distribution
gamma_shape = gamma_mean * gamma_mean / gamma_var  # Shape parameter of the Gamma distribution
gamma_scale = gamma_var / gamma_mean  # Scale parameter of the Gamma distribution
print(gamma_mean, gamma_var, gamma_shape, gamma_scale)
num_events = 600 * request_rate  # Number of events to generate

# Generate Poisson-distributed timestamps
poisson_timestamps = generate_poisson_timestamps(poisson_para, num_events)
save_timestamps_to_csv(poisson_timestamps, "poisson.csv")
analysis_request_rate(poisson_timestamps)

# Generate Gamma-distributed timestamps
gamma_timestamps = generate_gamma_timestamps(gamma_shape, gamma_scale, num_events)
save_timestamps_to_csv(gamma_timestamps, "./gamma.csv")
analysis_request_rate(gamma_timestamps)
