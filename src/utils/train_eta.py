import time


def format_time(seconds):
    days = seconds // (24 * 3600)
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{int(days)}d:{int(hours)}h:{int(minutes)}m:{int(seconds)}s"

def calculate_remaining_time(start_time, current_epoch, total_epochs):
    time_elapsed = time.time() - start_time # time per epoch
    remaining_epochs = total_epochs - current_epoch
    remaining_time = time_elapsed * remaining_epochs
    formatted_time = format_time(remaining_time)
    print(f"Epoch {current_epoch}/{total_epochs-1} - eta: {formatted_time}")
    
    return remaining_time