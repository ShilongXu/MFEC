# utils/Enhanced_early_stopper.py
"""
EnhancedEarlyStopper:
A robust early stopping utility for training loops.

Usage:
    stopper = EnhancedEarlyStopper(
        patience=10,
        min_epochs=20,
        improvement=0.01,
        stop_metric='composite',
        higher_is_better=True
    )

    for epoch in range(max_epochs):
        train_one_epoch()
        val_metrics = validate()

        # Update and check for improvement
        improved = stopper.update(epoch, val_metrics)
        if improved:
            save_best_model()
            print(f"New best {stopper.stop_metric}={val_metrics[stopper.stop_metric]:.4f} at epoch {epoch+1}")

        # Check if should stop
        if stopper.should_stop():
            print(f"Early stopping triggered at epoch {epoch+1}. No improvement for {stopper.patience} epochs.")
            break
"""


class EnhancedEarlyStopper:
    def __init__(
            self,
            patience: int = 10,
            min_epochs: int = 20,
            improvement: float = 0.01,
            stop_metric: str = 'total',
            higher_is_better: bool = False
    ):
        """
        Initializes the early stopper.

        Args:
            patience: Number of consecutive epochs without improvement to tolerate before stopping.
            min_epochs: Minimum number of epochs to run before early stopping can occur.
            improvement: Minimum relative improvement required to reset the patience counter.
            stop_metric: Name of the metric in validation metrics to monitor.
            higher_is_better: Whether a higher value of stop_metric is better.
        """
        self.patience = patience  # 容忍多少个 epoch 没有提升就停
        self.min_epochs = min_epochs  # 至少跑这么多 epoch 之后才考虑 early stop
        self.improvement = improvement  # 最小提升幅度
        self.stop_metric = stop_metric  # 比较的指标名（如 'composite'、'total'…）
        self.higher_is_better = higher_is_better

        # Initialize best score depending on whether higher is better
        if self.higher_is_better:
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

        self.counter = 0
        self.best_epoch = -1

    def update(self, epoch: int, val_metrics: dict) -> bool:
        """
        Call at the end of each epoch to update the stopper with the latest validation metric.

        Args:
            epoch: Current epoch index (0-based).
            val_metrics: Dictionary of validation metrics containing stop_metric.

        Returns:
            improved (bool): True if the monitored metric improved this epoch, False otherwise.
        """
        # Extract the score
        if self.stop_metric not in val_metrics:
            raise KeyError(f"Validation metrics must contain '{self.stop_metric}'")
        score = val_metrics[self.stop_metric]
        if not isinstance(score, (int, float)):
            raise TypeError(
                f"Validation metric '{self.stop_metric}' must be numeric, got {type(score)}"
            )

        improved = False
        # Before min_epochs, we update best_score but do not count toward patience
        if epoch < self.min_epochs:
            if self._is_improvement(score, self.best_score):
                self.best_score = score
                self.best_epoch = epoch
                improved = True
            return improved

        # After min_epochs, perform full early stopping logic
        if self._is_improvement(score, self.best_score):
            # Improvement found
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            improved = True
        else:
            # No improvement
            self.counter += 1

        return improved

    def should_stop(self) -> bool:
        """
        Returns True if the number of consecutive non-improving epochs
        has reached or exceeded the patience threshold.
        """
        return self.counter >= self.patience

    def _is_improvement(self, current: float, best: float) -> bool:
        """
        Internal helper to decide if current is an improvement over best
        based on the improvement threshold and direction.
        """
        if self.higher_is_better:
            return current > best + self.improvement
        else:
            return current < best - self.improvement
