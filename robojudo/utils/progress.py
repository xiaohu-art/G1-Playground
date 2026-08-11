from tqdm import tqdm


class ProgressBar:
    def __init__(self, tag, total):
        self.pbar = tqdm(
            total=total,
            desc=tag,
            unit="step",
            colour="magenta",
            ncols=100,
            dynamic_ncols=True,
            mininterval=0.01,
            leave=False,
            ascii=True,
        )

    def update(self, step=1):
        self.pbar.update(step)

    def close(self):
        self.pbar.close()
