"""
R1 parameter parity — size the MLP twin within ±15% of the KAN model (encoder+head),
report the comparison, and write counts to param_count.txt (R8).

Counts are closed-form (no module construction) so this is cheap for any input_dim.
report_parity RAISES if parity cannot be met ("do not proceed if parity violated").
"""
from __future__ import annotations

from pathlib import Path

PARITY_TOL = 0.15


def kan_layer_params(i: int, o: int, grid_size: int, spline_order: int) -> int:
    """base_weight (O*I) + spline_weight (O*I*(G+p))."""
    gp = grid_size + spline_order
    return o * i + o * i * gp


def kan_model_params(
    input_dim: int, hidden_dim: int, out_dim: int,
    n_enc_layers: int, grid_size: int, spline_order: int,
) -> int:
    p = kan_layer_params(input_dim, hidden_dim, grid_size, spline_order)
    for _ in range(n_enc_layers - 1):
        p += kan_layer_params(hidden_dim, hidden_dim, grid_size, spline_order)
    p += kan_layer_params(hidden_dim, out_dim, grid_size, spline_order)  # head
    return p


def mlp_linear_params(i: int, o: int) -> int:
    return i * o + o  # weight + bias


def mlp_model_params(input_dim: int, hidden_dim: int, out_dim: int, n_enc_layers: int) -> int:
    p = mlp_linear_params(input_dim, hidden_dim)
    for _ in range(n_enc_layers - 1):
        p += mlp_linear_params(hidden_dim, hidden_dim)
    p += mlp_linear_params(hidden_dim, out_dim)  # head
    return p


def matched_mlp_hidden(
    input_dim: int, out_dim: int, n_enc_layers: int, target_params: int,
    max_hidden: int = 8192,
) -> tuple[int, float]:
    """Scan hidden dim for the MLP total closest to target_params (monotone)."""
    best_h, best_ratio = 8, float("inf")
    for h in range(8, max_hidden):
        total = mlp_model_params(input_dim, h, out_dim, n_enc_layers)
        ratio = abs(total - target_params) / target_params
        if ratio < best_ratio:
            best_h, best_ratio = h, ratio
        elif total > target_params:
            break  # past the minimum
    return best_h, best_ratio


def parity_for_input_dim(
    input_dim: int, kan_hidden: int = 128, out_dim: int = 64,
    n_enc_layers: int = 2, grid_size: int = 5, spline_order: int = 3,
) -> dict:
    kan_total = kan_model_params(
        input_dim, kan_hidden, out_dim, n_enc_layers, grid_size, spline_order
    )
    mlp_hidden, ratio = matched_mlp_hidden(input_dim, out_dim, n_enc_layers, kan_total)
    mlp_total = mlp_model_params(input_dim, mlp_hidden, out_dim, n_enc_layers)
    return {
        "input_dim": input_dim, "kan_hidden": kan_hidden, "kan_total": kan_total,
        "mlp_hidden": mlp_hidden, "mlp_total": mlp_total, "ratio": ratio,
        "within_tol": ratio <= PARITY_TOL,
    }


def report_parity(
    input_dims: dict[str, int], kan_hidden: int = 128, out_dim: int = 64,
    n_enc_layers: int = 2, grid_size: int = 5, spline_order: int = 3,
    path: str | Path = "param_count.txt",
) -> list[dict]:
    """Compute parity per dataset, print + write param_count.txt, raise if violated."""
    rows = [
        {"name": name, **parity_for_input_dim(
            d, kan_hidden, out_dim, n_enc_layers, grid_size, spline_order)}
        for name, d in input_dims.items()
    ]
    header = (
        f"# KAN encoder L={n_enc_layers} hidden={kan_hidden}, head out={out_dim}, "
        f"grid G={grid_size}, spline_order p={spline_order}  (R1 parity, tol={PARITY_TOL:.0%})\n"
    )
    col = f"{'dataset':<12}{'in_dim':>8}{'KAN_params':>12}{'MLP_hidden':>12}{'MLP_params':>12}{'ratio':>9}{'ok':>5}\n"
    lines = [header, col, "-" * len(col) + "\n"]
    for r in rows:
        lines.append(
            f"{r['name']:<12}{r['input_dim']:>8}{r['kan_total']:>12}"
            f"{r['mlp_hidden']:>12}{r['mlp_total']:>12}{r['ratio']:>8.2%}"
            f"{'  yes' if r['within_tol'] else '   NO':>5}\n"
        )
    text = "".join(lines)
    Path(path).write_text(text, encoding="utf-8")
    print(text)

    violated = [r["name"] for r in rows if not r["within_tol"]]
    if violated:  # R9 / "do not proceed if parity violated"
        raise ValueError(
            f"R1 parameter parity violated (>{PARITY_TOL:.0%}) for: {violated}. "
            f"Adjust MLP search range or KAN width before proceeding."
        )
    return rows
