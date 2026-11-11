import atexit
import csv
import os
from tempfile import NamedTemporaryFile

class LossLogger:
    def __init__(self, out_dir, run_name, tensorboard_log, resume_step=None, resume_epoch=0):
        self.directory = os.path.join(out_dir, "metrics", run_name)
        os.makedirs(self.directory, exist_ok=resume_step is not None)
        for interval in ("step", "epoch"):
            path = self._csv_path(interval)
            with NamedTemporaryFile(mode="w", dir=self.directory, newline="", delete=False) as handle:
                temporary_path = handle.name
                writer = csv.writer(handle)
                writer.writerow(("step", "epoch", "train_loss", "val_loss"))
                if resume_step is not None and os.path.exists(path):
                    with open(path, newline="") as source:
                        for row in csv.DictReader(source):
                            if int(row["step"]) <= resume_step:
                                writer.writerow((row["step"], row["epoch"], row["train_loss"], row["val_loss"]))
            os.replace(temporary_path, path)
        print(f"Loss CSV files: {self.directory}")

        self.writers = {}
        if tensorboard_log:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                print("TensorBoard is unavailable. Install it with python -m pip install tensorboard.")
            else:
                log_dir = os.path.join(out_dir, "tensorboard", run_name)
                for interval, index in (("step", resume_step), ("epoch", resume_epoch)):
                    purge_step = index + 1 if resume_step is not None else None
                    writer = SummaryWriter(os.path.join(log_dir, interval), purge_step=purge_step, flush_secs=5)
                    self.writers[interval] = writer
                    atexit.register(writer.close)
                print(f"TensorBoard events: {log_dir}")

    def _csv_path(self, interval):
        return os.path.join(self.directory, f"{interval}s.csv")

    def record(self, interval, step, epoch, train_loss=None, val_loss=None):
        with open(self._csv_path(interval), "a", newline="") as handle:
            csv.writer(handle).writerow((step, epoch, train_loss, val_loss))
        if interval in self.writers:
            writer = self.writers[interval]
            index = step if interval == "step" else epoch
            if train_loss is not None:
                writer.add_scalar(f"{interval}/train_loss", train_loss, index)
            if val_loss is not None:
                writer.add_scalar(f"{interval}/val_loss", val_loss, index)