import numpy as np
import ipdb
import matplotlib.pyplot as plt
import os

def time_to_error():
    time_for_1 = []
    time_for_2 = []
    time_for_5 = []
    for i in range(1, 11):
        if i==4:
            continue
        errors = np.load(f'errors{i}.npy')
        times = np.load(f'times{i}.npy')
        time_for_5.append(times[np.argmax(errors<5)])
        time_for_2.append(times[np.argmax(errors<2)])
        time_for_1.append(times[np.argmax(errors<1)])
    ipdb.set_trace()

def time_to_error_hist():
    # folder = "landmarks/camera_ready/dets_and_poses_thres_fixed"
    folder = "landmarks/camera_ready/dets_and_poses_4orbits"
    # folder = "landmarks/camera_ready/orbits"
    # list all np arrays in the folder
    files = os.listdir(folder)
    # filter for npy files
    files = [f for f in files if f.endswith('.npy')]
    # sort the files
    files.sort()
    errors = []
    times = []
    for f in files:
        if 'errors' in f:
            errors+=list(np.load(folder + '/' + f, allow_pickle=True))
            times+=list(np.load(folder + '/' + f.replace('errors', 'times'), allow_pickle=True))
    # ipdb.set_trace()
    # errors = np.concatenate(errors)
    # times = np.concatenate(times)
    # errors = np.load(folder + '/errors.npy', allow_pickle=True)
    # times = np.load(folder + '/times.npy', allow_pickle=True)
    num_trajs = len(errors)
    time_for_10 = []
    time_for_5 = []
    time_for_2 = []
    time_for_1 = []

    for i in range(num_trajs):
        filtered_errors_10 = errors[i] < 10
        filtered_errors_5 = errors[i] < 5
        filtered_errors_2 = errors[i] < 2
        filtered_errors_1 = errors[i] < 1
        if np.any(filtered_errors_10):
            time_for_5.append(times[i][np.argmax(errors[i]<10)])
        else:
            pass
        if np.any(filtered_errors_5):
            time_for_5.append(times[i][np.argmax(errors[i]<5)])
        else:
            pass
        if np.any(filtered_errors_2):
            time_for_2.append(times[i][np.argmax(errors[i]<2)])
        else:
            pass
        if np.any(filtered_errors_1):
            time_for_1.append(times[i][np.argmax(errors[i]<1)])
        else:
            pass
            
    # ipdb.set_trace()
    time_for_10 = np.array(time_for_10)
    time_for_5 = np.array(time_for_5)
    time_for_2 = np.array(time_for_2)
    time_for_1 = np.array(time_for_1)
    # Sort the times for cumulative calculation
    time_for_10_sorted = np.sort(time_for_10)
    time_for_5_sorted = np.sort(time_for_5)
    time_for_2_sorted = np.sort(time_for_2)
    time_for_1_sorted = np.sort(time_for_1)
    
    # Calculate the cumulative fraction of trajectories
    # Normalize by the total number of trajectories, not just those filtered
    cumulative_fraction_10 = np.arange(1, len(time_for_10) + 1) / num_trajs
    cumulative_fraction_5 = np.arange(1, len(time_for_5) + 1) / num_trajs
    cumulative_fraction_2 = np.arange(1, len(time_for_2) + 1) / num_trajs
    cumulative_fraction_1 = np.arange(1, len(time_for_1) + 1) / num_trajs

    
    # Plotting the cumulative fraction
    plt.figure(figsize=(10, 6))
    # plt.step(time_for_10_sorted, cumulative_fraction_10, where='post', label='Fraction of Orbits <10km Error')
    plt.step(time_for_5_sorted, cumulative_fraction_5, where='post', label='Fraction of Orbits <5km Error')
    plt.step(time_for_2_sorted, cumulative_fraction_2, where='post', label='Fraction of Orbits <2km Error')
    plt.step(time_for_1_sorted, cumulative_fraction_1, where='post', label='Fraction of Orbits <1km Error')
    plt.title('Cumulative Fraction of First Times Reaching <xkm Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Fraction of Total Orbits')
    plt.ylim(0, 1)  # Ensure y-axis goes from 0 to 1 to represent the full range of fractions
    plt.grid(True)
    plt.legend()
    plt.savefig('time_to_x_orbits4.png')


if __name__ == '__main__':
    # time_to_error()
    time_to_error_hist()