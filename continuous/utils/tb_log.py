from tensorboard_logger import configure, log_value

def configure_tb(outdir):
    configure(outdir)

def log_and_print(key, value, t, multi=False):
    if multi:
        print("t:", t, end=" | ")
        for i in range(len(key)):
            end = " | " if i < len(key) - 1 else "\n"
            print("{}: {:.3f}".format(key[i], value[i]), end=end)
            log_value(key[i], value[i], t)    
    else:
        print("t:{}, {}: {:.3f}".format(t, key, value))
        log_value(key, value, t)
    