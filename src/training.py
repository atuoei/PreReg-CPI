# train.py
import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error

DEFAULTS = dict(
    lr=1e-3,
    weight_decay=1e-5,
    epochs=100,
    patience=15,
    step_size=20,
    gamma=0.5,
)

def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(x, device) for x in obj)
    return obj

def _unpack_batch(batch):
    """
    支持：
      (genes, (x1,x2), y)
      (genes, x1, x2, y)
      (x1, x2, y)
      (genes, x, y)
      (x, y)
    返回: inputs(list/tuple), y(tensor), genes(list或None)
    """
    b = list(batch)
    y = b[-1]
    genes = None
    start = 0
    if len(b) >= 3 and isinstance(b[0], (list, tuple)) and b[0] and isinstance(b[0][0], str):
        genes = b[0]
        start = 1
    inputs = b[start:-1]
    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        inputs = list(inputs[0])
    return inputs, y, genes

class Trainer:
    def __init__(self, cfg=None, mode: str = 'regression'):
        """
        mode: 'classification' 或 'regression'
        """
        self.cfg = {**DEFAULTS, **(cfg or {})}
        m = str(mode).lower()
        assert m in ('classification', 'regression'), "mode must be 'classification' or 'regression'"
        self.mode = m
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- loss -------
    def _loss_and_metrics(self, y_pred, y_true):
        if self.mode == 'classification':
            # BCEWithLogitsLoss + AUC
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)
            y_prob = torch.sigmoid(y_pred.detach()).cpu().numpy().ravel()
            yt = y_true.detach().cpu().numpy().ravel()
            if len(np.unique(yt)) > 1:
                auc = roc_auc_score(yt, y_prob)
            else:
                auc = float("nan")
            return loss, {"AUC": auc}, "AUC"
        else:
            loss = torch.nn.functional.mse_loss(y_pred, y_true)
            yp = y_pred.detach().cpu().numpy().ravel()
            yt = y_true.detach().cpu().numpy().ravel()
            mse = mean_squared_error(yt, yp)
            return loss, {"R2": r2_score(yt, yp), "MSE": mse, "RMSE": math.sqrt(mse)}, "R2"

    # ------- train-------
    def fit(self, model, train_loader, val_loader, save_path):
        model = model.to(self.device)
        optim = torch.optim.AdamW(model.parameters(),
                                  lr=self.cfg["lr"],
                                  weight_decay=self.cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.StepLR(optim, self.cfg["step_size"], self.cfg["gamma"])

        best, bad = -np.inf, 0
        for epoch in range(self.cfg["epochs"]):
            # train
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                optim.zero_grad()
                inputs, y, _ = _unpack_batch(batch)
                inputs = [_to_device(x, self.device) for x in inputs]
                if self.mode == 'classification':
                    y = _to_device(y, self.device).float().view(-1, 1)
                else:
                    y = _to_device(y, self.device).float().view(-1, 1)
                
                y_pred = model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
                if self.mode == 'classification' and y_pred.dim() == 1:
                    y_pred = y_pred.view(-1, 1)
                loss, _, _ = self._loss_and_metrics(y_pred, y)
                loss.backward()
                optim.step()
                total_loss += float(loss)
            sched.step()

            # validate
            model.eval()
            yps, yts = [], []
            with torch.no_grad():
                for batch in val_loader:
                    inputs, y, _ = _unpack_batch(batch)
                    inputs = [_to_device(x, self.device) for x in inputs]
                    y = _to_device(y, self.device).float().view(-1, 1)
                    yp = model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
                    if self.mode == 'classification' and yp.dim() == 1:
                        yp = yp.view(-1, 1)
                    yps.append(yp.detach().cpu())
                    yts.append(y.detach().cpu())
            yps = torch.cat(yps, dim=0); yts = torch.cat(yts, dim=0)
            _, metrics, key = self._loss_and_metrics(yps, yts)
            avg_loss = total_loss / max(1, len(train_loader))
            print(f"[{self.mode}] epoch {epoch+1}/{self.cfg['epochs']} | loss={avg_loss:.4f} | {key}={metrics.get(key, float('nan')):.4f}")

            score = metrics.get(key, float("-inf"))
            if np.isfinite(score) and score > best:
                best, bad = score, 0
                torch.save(model.state_dict(), save_path)
                print("  ↳ saved:", save_path)
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    print("  ↳ early stop.")
                    break

    # ------- test -------
    def test(self, model, test_loader):
        model = model.to(self.device)
        model.eval()
        genes_all, y_all, yp_all = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs, y, genes = _unpack_batch(batch)
                inputs = [_to_device(x, self.device) for x in inputs]
                y = _to_device(y, self.device).float().view(-1, 1)
                yp = model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
                if self.mode == 'classification' and yp.dim() == 1:
                    yp = yp.view(-1, 1)
                if genes is not None:
                    genes_all.extend(list(genes))
                y_all.append(y.detach().cpu())
                yp_all.append(yp.detach().cpu())
        y_all = torch.cat(y_all, dim=0)
        yp_all = torch.cat(yp_all, dim=0)
        _, metrics, _ = self._loss_and_metrics(yp_all, y_all)
        return genes_all, y_all.numpy().ravel().tolist(), yp_all.numpy().ravel().tolist(), metrics
