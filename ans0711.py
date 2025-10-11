import tensorflow as tf
import os
import re

# Define the paths
log_file_path = "/home/geofly/pa-sam-Hao/output/AKres/Kres_train.txt"
tensorboard_log_dir = "logs/fit"

# Create the directory for TensorBoard logs if it doesn't exist
os.makedirs(tensorboard_log_dir, exist_ok=True)

# Initialize summary writers for TensorBoard
summary_writer = tf.summary.create_file_writer(tensorboard_log_dir)

# Define regex patterns to extract values
patterns = {
    'step': re.compile(r'INFO:root:step: +(\d+)'),
    'training_loss': re.compile(r'training_loss: +([0-9.]+)'),
    'loss_mask': re.compile(r'loss_mask: +([0-9.]+)'),
    'loss_dice': re.compile(r'loss_dice: +([0-9.]+)'),
    'loss_mask_final': re.compile(r'loss_mask_final: +([0-9.]+)'),
    'loss_dice_final': re.compile(r'loss_dice_final: +([0-9.]+)'),
    'loss_uncertain_map': re.compile(r'loss_uncertain_map: +([0-9.]+)'),
    'val_iou': re.compile(r'val_iou: +([0-9.]+)')
}

# Parse the log file and write to TensorBoard
with open(log_file_path, 'r') as f:
    step = 0
    for line in f:
        for metric, pattern in patterns.items():
            match = pattern.search(line)
            if match:
                value = float(match.group(1))
                with summary_writer.as_default():
                    tf.summary.scalar(metric, value, step=step)
        step += 1

# Close the summary writer
summary_writer.close()

print("TensorBoard logs have been written to", tensorboard_log_dir)

