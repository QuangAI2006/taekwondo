"""Reuse the scoring pipeline with a timed replay for sequential batch processing."""
import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch-preview-seconds", type=float, default=3.0)
    opts, remaining = parser.parse_known_args()
    if opts.batch_preview_seconds < 0:
        parser.error("--batch-preview-seconds must be non-negative")
    sys.argv = [sys.argv[0], *remaining]
    import matplotlib.pyplot as plt
    import pipeline_viewer

    class BatchPipelineViewer(pipeline_viewer.PipelineViewer):
        def _start_replay(self):
            # A video without a detected person has frames but no pose sequence.
            if self.kpts is None or len(self.kpts) == 0:
                return
            super()._start_replay()

        def wait_close(self):
            deadline = time.monotonic() + opts.batch_preview_seconds
            while (self.interactive and not self.closed and plt.fignum_exists(self.fig.number)
                   and time.monotonic() < deadline):
                plt.pause(0.1)
            plt.close(self.fig)

    pipeline_viewer.PipelineViewer = BatchPipelineViewer
    from infer_action import main as infer_main
    infer_main()


if __name__ == "__main__":
    main()
