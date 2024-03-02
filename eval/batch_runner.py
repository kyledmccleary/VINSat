import subprocess

regions = ['10S', '10T', '11R', '12R', '16T', '17R', #'17T', '18S', 
           '32S', '32T', '33S', '33T', '52S', '53S', '54S', '54T']

for region in regions:    
    # Define the command you want to execute
    # command = "python eval_landmarks.py --model_path ../sim/models/" + region + ".pt --im_path test_dataset/" + region + "/images --lab_path test_dataset/" + region + "/labels --output_path " + region + "_err.npy --calculate_err --save_err"
    command = "python eval_landmarks.py --err_path " + region + "_err.npy --best_classes --save_best_conf --best_classes_path ../sim/best_classes/" + region + "_best_classes.npy --best_conf_path ../sim/best_confs/" + region + "_best_conf.npy --px_threshold 10" 
    # Execute the command in the command line
    subprocess.call(command, shell=True)
