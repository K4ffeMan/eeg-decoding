import torch
import torch.nn as nn


class RSynaptic(nn.Module):
    """A recurrent spiking neuron with separate synaptic-current and membrane states."""

    def __init__(
        self,
        in_features,
        alpha=0.9,
        beta=0.85,
        threshold=1.0,
        recurrent_strength=1.0,
        reset_strength=1.0,
        surrogate_slope=25.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.recurrent_strength = recurrent_strength
        self.reset_strength = reset_strength
        self.surrogate_slope = surrogate_slope

        self.recurrent = nn.Linear(in_features, in_features, bias=False)

    def forward(self, input_current, prev_spike, synaptic_current, membrane):
        synaptic_current = (
            self.alpha * synaptic_current
            + self.recurrent_strength * self.recurrent(prev_spike)
            + input_current
        )
        membrane = self.beta * membrane + synaptic_current

        membrane_for_spike = membrane
        hard_spike = (membrane_for_spike >= self.threshold).to(membrane.dtype)

        # Fast sigmoid surrogate: exact hard spikes in the forward pass, smooth gradient in backprop.
        sigmoid = torch.sigmoid(
            self.surrogate_slope * (membrane_for_spike - self.threshold)
        )
        spike = (hard_spike - sigmoid).detach() + sigmoid

        membrane = membrane - self.reset_strength * self.threshold * hard_spike
        return spike, synaptic_current, membrane


class Net(nn.Module):
    def __init__(
        self,
        n_in=256,
        n_hidden=128,
        n_out=4,
        alpha=0.9,
        beta=0.85,
    ):
        super().__init__()
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(n_in, n_hidden)
        self.hidden = RSynaptic(n_hidden, alpha=alpha, beta=beta)
        self.fc_out = nn.Linear(n_hidden, n_out)

        # Membrane-only output state for stable, continuous 4-class logits.
        self.output_decay = beta

    def forward(self, x):
        """Classify spike trains shaped [batch, time, 256]."""
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, 256]")

        batch_size, num_steps, n_in = x.shape
        if n_in != self.fc1.in_features:
            raise ValueError(f"expected {self.fc1.in_features} input features, got {n_in}")

        device = x.device
        dtype = x.dtype
        synaptic = x.new_zeros(batch_size, self.n_hidden)
        membrane = x.new_zeros(batch_size, self.n_hidden)
        spike = x.new_zeros(batch_size, self.n_hidden)
        output_membrane = x.new_zeros(batch_size, self.fc_out.out_features, device=device, dtype=dtype)
        output_membranes = []

        for t in range(num_steps):
            synaptic_input = self.fc1(x[:, t])
            spike, synaptic, membrane = self.hidden(
                synaptic_input,
                spike,
                synaptic,
                membrane,
            )

            output_current = self.fc_out(spike)
            output_membrane = self.output_decay * output_membrane + output_current
            output_membranes.append(output_membrane)

        if not output_membranes:
            raise ValueError("the input time dimension must contain at least one timestep")

        return torch.stack(output_membranes, dim=0).mean(dim=0)
