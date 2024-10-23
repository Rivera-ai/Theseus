import torch
import numpy as np

def NormalKl(mean1, logvar1, mean2, logvar2):
    """
        Calcula la divergencia KL entre dos gaussianas.
        Las formas se transmiten automáticamente, por lo que los lotes se pueden comparar con
        escalares, entre otros casos de uso.
    """

    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "Al menos un argumento debe ser un tensor."

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
        Una aproximación rápida de la función de distribución acumulativa de la
        normal estándar.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def continuous_gaussian_log_likelihood(x, *, means, log_scales):
    """
        Calcula la verosimilitud logarítmica de una distribución gaussiana continua.
        :param x: los objetivos
        :param means: el tensor de media gaussiana.
        :param log_scales: el tensor de desviación estándar logarítmica gaussiana.
        :return: un tensor como x de probabilidades logarítmicas (en valores naturales).
    """
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    normalized_x = centered_x * inv_stdv
    log_probs = torch.distributions.Normal(torch.zeros_like(x), torch.ones_like(x)).log_prob(normalized_x)
    return log_probs

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
        Calcula la verosimilitud logarítmica de una distribución gaussiana discretizándola a una
        imagen dada.
        :param x: las imágenes de destino. Se supone que se trata de valores uint8,
            reescalados al rango [-1, 1].
        :param means: el tensor de media gaussiana.
        :param log_scales: el tensor de desviación estándar logarítmica gaussiana.
        :return: un tensor como x de probabilidades logarítmicas (en nats).
    """

    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where( x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs