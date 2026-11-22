import torch

from ml_training_lab.diagnostics import gradient_step


def test_analytical_gradients_and_sgd_update():
    result = gradient_step()
    for values in result.values():
        torch.testing.assert_close(values["autograd"], values["analytical"])
        torch.testing.assert_close(values["after"], values["before"] - 0.1 * values["analytical"])
    torch.testing.assert_close(result["user"]["autograd"], torch.tensor([-0.25, -0.125], dtype=torch.float64))
    torch.testing.assert_close(result["movie"]["autograd"], torch.tensor([-0.125, 0.25], dtype=torch.float64))
