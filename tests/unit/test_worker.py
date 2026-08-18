from cortaflow.workers.base_worker import FunctionWorker


def test_worker_cooperative_cancellation() -> None:
    worker = FunctionWorker(lambda progress, cancelled: cancelled.is_set())
    worker.cancel()
    results = []
    worker.signals.finished.connect(results.append)
    worker.run()
    assert results == [True]

