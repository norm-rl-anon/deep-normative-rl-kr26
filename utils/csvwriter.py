import csv


class StatsWriter:
    def __init__(self, study_name, trial_number):
        self.filename = f"{study_name}_{trial_number:03}.csv"
        self.writer = None

    def _write_line(self, line):
        if self.writer is None:
            self.file = open(self.filename, mode="w")
            self.writer = csv.DictWriter(self.file, fieldnames=sorted(line))
            self.writer.writeheader()
        self.writer.writerow(line)
        self.file.flush()

    def write_trial(self, monitors, extra):
        row = {}
        for monitor in monitors:
            row.update(monitor.export())
        row.update(extra)
        self._write_line(row)
